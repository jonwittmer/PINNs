#!/usr/bin/env python3
"""
@author: Maziar Raissi
Modified by Hwan Goh, Oden Institute of Computational Sciences and Engineering, 10/8/2019
This code is associated with the paper: "Physics Informed Deep Learning (Part I)- Data-driven solutions of nonlinear partial differential equations".
For regularization by the residual, we consider the L^1-norm instead of the L^2-norm. Optimization is done using the alternating direction method of multipliers (ADMM)
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

np.random.seed(1234)
tf.set_random_seed(1234)

###############################################################################
#                  Construct Physics Induced Neural Network                   #
###############################################################################
class PhysicsInformedNN:
    ############################
    #   Initialize the Class   #
    ############################
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, lagrange_initial_guess, penalty_parameter, filename, GPU_number): # "__init__" is a constructor for a class, it is called automatically if you create an instance of a class
        
        
        self.filename = filename # for book-keeping purposes
        
        #=== Spatial and Temporal Domain Attributes ===#
        self.lb = lb
        self.ub = ub
    
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.u = u
        
        self.layers = layers
        self.nu = nu
        
        #=== Number of Data Points and Number of Collocation Points ===#
        self.N_u = self.x_u.shape[0]
        self.N_r = self.x_f.shape[0]
        
        #=== Initialize NNs ===#
        self.weights, self.biases = self.initialize_NN(layers)
        
        #=== tf placeholders and Graph ===#
        self.gpu_options = tf.GPUOptions(visible_device_list=GPU_number)

        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True,
                                     intra_op_parallelism_threads=1,
                                     inter_op_parallelism_threads=4,
                                     gpu_options=self.gpu_options)

        
        self.sess = tf.Session(config=self.config) # GPU related stuff
        
        #=== Initialize Data Variables ===#
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]]) # By setting it as "None", it allows any size. This usually corresponds to the batch size which can vary
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        #=== Initialize Collocation Variables ===#
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])        
        
        #=== Forward Prediction of u and Residual ===#        
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)         
        
        #=== Lagrange Multiplier and z Dummy Variable ===#
        self.lagrange = tf.Variable(tf.ones([self.N_r,1]), dtype=tf.float32,trainable = False)
        self.z = tf.Variable(tf.ones([self.N_r,1]), dtype=tf.float32,trainable = False) # z will later be initialized to equal r(w) after all the variables of the graph is intialized. However, we create it here first to ensure the dimensions are correct
        self.rho = tf.constant(penalty_parameter)

        #=== Updating Lagrange Multiplier and z Dummy Variable ===#
        self.one_over_rhoN_r = 1/(self.rho*self.N_r)
        self.z_update = self.z.assign(self.soft_thresholding())        
        self.lagrange_update = self.lagrange.assign(self.lagrange + self.rho*(self.f_pred - self.z))
               
        #=== Lagrange Multiplier, Loss Function and Optimizer ===#        
        self.loss = (1/self.N_u)*tf.pow(tf.norm(self.u_tf - self.u_pred,2),2) + \
                    tf.matmul(tf.transpose(self.lagrange), self.net_f(self.x_f_tf, self.t_f_tf)) + \
                    (self.rho/2)*tf.pow(tf.norm(self.f_pred - self.z + tf.div(self.lagrange,self.rho),2),2)
                                   
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
        self.tol = 0.0001
        
        self.lagrange = self.lagrange.assign(self.lagrange + self.rho * (self.f_pred - self.z))
        self.z_update = self.z.assign(self.soft_thresholding())
    
        #=== Initialize and Run ===#
        init = tf.global_variables_initializer() # by "model = PhysicsInformedNN" in the driver, a Tensorflow session has been constructed and initialized in __init__ of this class
        self.sess.run(init)
        
        #=== Assign the real Initial value of z = r(w) ===#
        self.sess.run(self.z.assign(self.f_pred), 
                      feed_dict={self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}) # Here, z is initialized to equal to r(w). This only happens once: when an instance of PhysicsInformedNN is created

    ####################################
    #   Initialize Weights and Biases  #
    ####################################            
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
    
    ######################################
    #   Initialize Weights Using Xavier  #
    ######################################        
    def xavier_init(self, size): # Xavier is a specific choice of the initial weight values
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    #################################
    #   Neural Network Operations   #
    #################################
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 # Scaled Input Values
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b)) # H = tanh(W*H + b)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    ###############################################################################
    #   Forward Propagation of u Neural Network and PDE Residual Neural Network   #
    ###############################################################################
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x,t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - self.nu*u_xx   
        return f
    
    ################################################
    #   Callback for Displaying Training Outputs   #
    ################################################
    def callback(self, loss):
        print('Loss:', loss)

    ############################
    #   Train Neural Network   #
    ############################ 
    def train(self,number_of_ADMM_iterations,number_of_w_optimization_steps,filename,GPU_number):        
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
                                                                                                                          
        start_time = time.time()
        #=== Iterations ===#
        iter_counter = 0
        loss_value = 1000
        while iter_counter < number_of_ADMM_iterations and abs(loss_value) > self.tol:
            self.sess.run(self.train_op_Adam, tf_dict)
            #=== Set Number of Iterations to Optimize w ===#
            if iter_counter % number_of_w_optimization_steps == 0:
                self.sess.run(self.z_update, tf_dict)
                self.sess.run(self.lagrange_update, tf_dict)                    
            #=== Print Iteration Information ===#
            if iter_counter % 100 == 0:
                time_elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('%s: \nIteration Number: %d, Loss: %.3e, Time Elapsed: %.2f, GPU Number: %s\n' %(filename, iter_counter, loss_value, time_elapsed, GPU_number))
                start_time = time.time()            
            iter_counter += 1
        
    ###################################
    #   Soft-Thresholding Operation   #
    ###################################
    def soft_thresholding(self):
        val = self.f_pred + self.lagrange/self.rho
        # annoying digital logic workaround to implement conditional.
        # construct vectors of 1's and 0's that we can multiply
        # by the proper value and sum together
        cond1 = tf.where(tf.greater(val, self.one_over_rhoN_r), tf.ones([self.N_r,1]), tf.zeros([self.N_r,1]))
        cond3 = tf.where(tf.less(val, -1*self.one_over_rhoN_r), tf.ones([self.N_r,1]), tf.zeros([self.N_r,1]))
        # cond2 is not needed since the complement of the intersection
        # of (cond1 and cond3) is cond2 and already assigned to 0
        z = cond1*(val - 1/(self.rho*self.N_r)) + cond3*(val + 1/(self.rho*self.N_r))        
        return z
                                    
    ####################################
    #   Predict Using Neural Network   #
    ####################################
    def predict(self, X_star):                
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})               
        return u_star, f_star

###############################################################################
#                                 Driver                                      #
###############################################################################    
if __name__ == "__main__": 
    
    #######################################
    #   Construct Space and Time Domain   #
    #######################################
    nu = 0 # Value of lambda_2   
    noise = 0.0        

    N_u = 100 # number of training points
    N_f = 10000 # number of collocation points
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t = data['t'].flatten()[:,None] # 100 time points
    x = data['x'].flatten()[:,None] # 256 spatial points
    Exact = np.real(data['usol']).T # 100 x 256 array of true values at every spatial point and at every time step
    
    X, T = np.meshgrid(x,t) # Repeats X row wise 100 times and repeats T column wise 256 times to form 100 x 256 array
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # Forms 25600 x 2 array which associates each set of 256 spatial points (column 1) to one time point (column 2)
    u_star = Exact.flatten()[:,None]              
    
    #=== Domain bounds ===#
    lb = X_star.min(0) # First spatial point at earliest time step
    ub = X_star.max(0) # Last spatial point at latest time step  
    
    #######################################
    #   Initial and Boundary Conditions   #
    #######################################    
    #=== Initial Condition ===#
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # 256 x 2 matrix where first column is all the spatial values x and the second column is all zeros, representing time=0
    uu1 = Exact[0:1,:].T # 256 x 1 matrix of initial condition values: takes the first row of Exact which contains the values at all 256 spatial points at time=0
    
    #=== Boundary Conditions ===#
    xx2 = np.hstack((X[:,0:1], T[:,0:1])) # 100 x 2 matrix where first column is all -1 which represents the left boundary points and the second column is all the time points
    uu2 = Exact[:,0:1] # 100 x 2 matrix of left boundary condition values: takes the first column of Exact which contains the values at x=-1 at all 100 time points
    xx3 = np.hstack((X[:,-1:], T[:,-1:])) # 100 x 2 matrix where first column is all -1 which represents the left boundary points and the second column is all the time points
    uu3 = Exact[:,-1:] # 100 x 2 matrix of right boundary condition values: takes the last column of Exact which contains the values at x=-1 at all 100 time points
    
    X_u_train = np.vstack([xx1, xx2, xx3]) # 456 x 2 matrix which is the stacked initial and boundary conditions
    u_train = np.vstack([uu1, uu2, uu3]) # 456 x 1 matrix of stacked initial and boundary condition values
    X_f_train = lb + (ub-lb)*lhs(2, N_f) # 10000 x 2 matrix. Randomly selected collocation points. lhs is a Latin hypercube which creates random arrows. For example, lhs(x,1) creates a row (why row?) array of numbers between 0 and 1
    X_f_train = np.vstack((X_f_train, X_u_train)) # 10456 x 2 matrix stacking the initial and boundary conditions with the collocation points
      
    ##############################
    #   Construct Training Data  #
    ##############################
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False) # Out of the array [0,1,2,...,X_star.shape[0]], select N_u random numbers without repeats. That is, once a number is select it, remove it from the array and do not replace it
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
    
    ###########################
    #   Training Parameters   #
    ###########################       
    lagrange_initial_guess = 1
    penalty_parameter = 0.5
    number_of_ADMM_iterations = 1
    number_of_w_optimization_steps = 1
    GPU_number = '3'
    
    ############################################
    #   Construct, Train and Run PINNs Model   #
    ############################################    
    #=== Filename ===#
    filepath = 'figures/'
    penalty_parameter_string = '%.2f' %(penalty_parameter)
    penalty_parameter_string_after_decimal = penalty_parameter_string.split('.',1)
    filename = 'L1ADMM_0%sP_%dT_%dC_%dE_%dwMinE' %(penalty_parameter_string_after_decimal[1], N_u, N_f, number_of_ADMM_iterations, number_of_w_optimization_steps)
            
    #=== Construct PINNs ===#
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, lagrange_initial_guess, penalty_parameter, filename, GPU_number)
    
    #=== Begin Training ===#
    start_time = time.time()                
    model.train(number_of_ADMM_iterations, number_of_w_optimization_steps, filename, GPU_number)
    elapsed = time.time() - start_time                
    print('Total Training time: %.4f' %(elapsed))
    
    #=== Model Prediction and Error ===#
    u_pred, f_pred = model.predict(X_star)           
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Error u: %e' %(error_u))                        
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    
    
    ###########################################################################
    #                               Plotting                                  #
    ###########################################################################    
    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    #=== Row 0: u(t,x) ===#   
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    #=== Row 1: u(t,x) slices ===#    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.set_title('$t = 0$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([0, np.pi])
    ax.set_ylim([0, 0.7])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.axis('square')
    ax.set_xlim([0, np.pi])
    ax.set_ylim([0, 0.7])
    ax.set_title('$t = 0.50$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[-1, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[-1, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.axis('square')
    ax.set_xlim([0, np.pi])
    ax.set_ylim([0, 0.7])
    ax.set_title('$t=3.14$', fontsize=10)
    
    # Saving Figure    
    print('\nFigure saved to ' + filepath + filename)
    plt.savefig(filepath + filename, dpi=300)

    



