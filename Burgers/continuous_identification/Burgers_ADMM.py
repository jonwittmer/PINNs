#!/usr/bin/env python3
"""
@author: Maziar Raissi
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '../../Utilities/')


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_data, u_data, layers, lb, ub, X_phys=[]):
        
        self.lb = lb
        self.ub = ub
        
        # break up data
        self.x_data = X_data[:, 0:1]
        self.t_data = X_data[:, 1:2]
        self.u = u_data

        # break up physics terms. If no physics terms are provides, default to using the data
        if len(X_phys) == 0:
            self.x_phys = X_data[:, 0:1]
            self.t_phys = X_data[:, 1:2]
        else:
            self.x_phys = X_phys[:, 0:1]
            self.t_phys = X_phys[:, 1:2]
            
        # Save dimensions
        self.N_u = self.x_data.shape[0]
        self.N_r = self.x_phys.shape[0]

        # list of number of nodes in each layer - first is input layer
        self.layers = layers
        
        # Initialize training variables
        self.weights, self.biases = self.initialize_NN(layers)

        # set configuration options
        self.gpu_options = tf.GPUOptions(visible_device_list='3')
        
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True,
                                     intra_op_parallelism_threads=4,
                                     inter_op_parallelism_threads=2,
                                     gpu_options=self.gpu_options)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=self.config)
        
        # Initialize PDE parameters
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32, trainable=False)
        self.lambda_2 = tf.Variable([0.0031831], dtype=tf.float32, trainable=False)
        
        # placeholders for training data
        self.x_data_tf = tf.placeholder(tf.float32, shape=[None, self.x_data.shape[1]])
        self.t_data_tf = tf.placeholder(tf.float32, shape=[None, self.t_data.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
                
        # placeholders for training collocation points
        self.x_phys_tf = tf.placeholder(tf.float32, shape=[None, self.x_phys.shape[1]])
        self.t_phys_tf = tf.placeholder(tf.float32, shape=[None, self.t_phys.shape[1]])

        # evaluate outputs of network
        self.u_pred = self.net_u(self.x_data_tf, self.t_data_tf)
        self.f_pred = self.net_f(self.x_phys_tf, self.t_phys_tf)

        # initialize the ADMM variables - just lists here as tensorflow isn't training them
        self.z = tf.Variable(tf.ones([self.N_r, 1]), dtype=tf.float32, trainable=False)
        self.gamma = tf.Variable(tf.ones([self.N_r, 1]), dtype=tf.float32, trainable=False)
        self.rho = tf.constant(0.5)
        self.c_gamma = 1 / (self.rho * self.N_r)
        self.zeros = tf.zeros((self.N_r, 1))
        self.ones  = tf.ones((self.N_r, 1))

        # ADMM loss term for training the weights - use backprop on this
        self.loss = 1 / self.N_u * tf.pow(tf.norm(self.u - self.u_pred, 2), 2) + \
                    self.rho / 2 * tf.pow(tf.norm(self.f_pred - self.z + self.gamma / self.rho, 2), 2)
            
        self.gamma_update = self.gamma.assign(self.gamma + self.rho * (self.f_pred - self.z))
        self.z_update = self.z.assign(self.compute_z())
        
        self.admm_misfit = tf.reduce_mean(tf.abs(self.f_pred - self.z))
        self.tol = 0.0001
        '''
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100,
                                                                           'maxfun': 100,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol': 100.0 * np.finfo(float).eps})
        '''
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 
                                                          #var_list=[self.weights, 
                                                          #          self.biases])
                                                                    #self.lambda_1,
                                                                    #self.lambda_2])
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # assign the real initial value of z = r(w) 
        self.sess.run(self.z.assign(self.f_pred), 
                      feed_dict={self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys})

        self.data_frame = pd.DataFrame()

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u
    
    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        #lambda_2 = tf.exp(self.lambda_2)
        lambda_2 = self.lambda_2
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        
        return f
    
    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, lambda_2))
        
    def compute_z(self):
        val   = self.f_pred + self.gamma / self.rho

        # annoying digital logic workaround to implement conditional.
        # construct vectors of 1's and 0's that we can multiply
        # by the proper value and sum together
        cond1 = tf.where(tf.greater(val, self.c_gamma), self.ones, self.zeros)
        cond3 = tf.where(tf.less(val, -1.0 * self.c_gamma), self.ones, self.zeros)
        # cond2 is not needed since the complement of the intersection
        # of (cond1 and cond3) is cond2 and already assigned to 0

        dummy_z = cond1 * (val - self.c_gamma) + cond3 * (val + self.c_gamma)
        
        return dummy_z

    def train(self, nIter):
        tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.u_tf: self.u, 
                   self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}
        
        # main iterations: updating Lagrange multiplier

        start_time = time.time()
        it = 0
        loss_value = 1000

        while it < nIter and abs(loss_value) > self.tol:
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # only update every 10 iterations
            if it % 1 == 0:
                self.sess.run(self.z_update, tf_dict)
                self.sess.run(self.gamma_update, tf_dict)
                    
            # print to monitor results
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                r_z = self.sess.run(self.admm_misfit, tf_dict)
                print('It: %d, Loss: %.3e, r(w) - z: %.3f ,Time: %.2f' %
                      (it, loss_value, r_z, elapsed))
                start_time = time.time()
            
            # save figure every so often so if it crashes, we have some results
            if it % 10000 == 0:
                self.plot_results()
            
            it += 1

    def predict(self, X_star):
        
        tf_dict = {self.x_data_tf: X_star[:, 0:1], self.t_data_tf: X_star[:, 1:2],
                   self.x_phys_tf: X_star[:, 0:1], self.t_phys_tf: X_star[:, 1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star
        
    def plot_results(self):
        pass

    
if __name__ == "__main__":
     
    nu = 0.01 / np.pi

    N_u = 100
    N_r = 100000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
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
    #X_f_train = lb + (ub-lb)*lhs(2, N_f) # 10000 x 2 matrix. Randomly selected collocation points. lhs is a Latin hypercube which creates random arrows. For example, lhs(x,1) creates a row (why row?) array of numbers between 0 and 1
    #X_f_train = np.vstack((X_f_train, X_u_train)) # 10456 x 2 matrix stacking the initial and boundary conditions with the collocation points
      
    ##############################
    #   Construct Training Data  #
    ##############################
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False) # Out of the array [0,1,2,...,X_star.shape[0]], select N_u random numbers without repeats. That is, once a number is select it, remove it from the array and do not replace it
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    #noise = 0.0
             
    #idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    #X_u_train = X_star[idx, :]
    #u_train = u_star[idx, :]
    X_phys_train = np.random.uniform(lb, ub, (N_r,2))
    
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, X_phys_train)
    model.train(1000000)
    
    u_pred, f_pred = model.predict(X_star)
            
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    #lambda_2_value = np.exp(lambda_2_value)
    
    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100
    
    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    '''
    noise = 0.01
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
        
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(10000)
    
    u_pred, f_pred = model.predict(X_star)
        
    lambda_1_value_noisy = model.sess.run(model.lambda_1)
    lambda_2_value_noisy = model.sess.run(model.lambda_2)
    lambda_2_value_noisy = np.exp(lambda_2_value_noisy)
            
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))
    '''
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    plt.rc('text', usetex=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1.0 / 3.0 + 0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=2, clip_on=False)
    
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize=10)
    
    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=10)
    
    ####### Row 3: Identified PDE ##################
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0 - 2.0 / 3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)
    
    ax = plt.subplot(gs2[:, :])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'\\ \hline' #r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1 + s2 + s3 + s4 + s5
    ax.text(0.1, 0.1, s)
    
    filename = 'figures/Correct/Train_1e6/ADMM_Z_1_Nr_100.png'
    print()
    print('Figure saved to ' + filename)
    print()

    plt.savefig(filename, dpi=300)
    #plt.show()
