#!/usr/bin/env python3
"""
@author: Maziar Raissi
@author: Jon Wittmer
@author: Hwan Goh
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
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')


np.random.seed(1234)
tf.set_random_seed(1234)


class Parameters:
    N_data = 200
    N_f    = 1000
    pen    = 40.0
    epochs = 1e5
    gpu    = '3'


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, params):
        
        self.params = params
        self.load_data()

        self.tol = 1e-4
        
        # Save dimensions
        self.N_data = self.params.N_data
        self.N_f = self.params.N_f
       
        # Initialize training variables
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.initialize_variables()

        # Evaluate outputs of network
        self.rho_pred, self.u_pred. self.E_pred = self.net_rho_u_E(self.x_data_tf, self.t_data_tf)
        self.f1_pred, self.f2_pred, self.f3_pred = self.net_f(self.x_phys_tf, self.t_phys_tf)
        
        # Initialize ADMM variables
        self.initialize_ADMM()
        
        # Optimizer
        self.optimizer_Adam  = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam   = self.optimizer_Adam.minimize(self.loss)
        

        # Set GPU configuration options
        self.gpu_options = tf.GPUOptions(visible_device_list=self.params.gpu,
                                         allow_growth=True)        
        self.config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=True,
                                     intra_op_parallelism_threads=4,
                                     inter_op_parallelism_threads=2,
                                     gpu_options=self.gpu_options)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=self.config)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # Randomly choose collocations points
        self.x_phys = np.random.uniform(self.lb[0], self.ub[0], [self.params.N_f,1] )
        self.t_phys = np.random.uniform(self.lb[1], self.ub[1], [self.params.N_f,1])

        # Assign the real initial value of z = r(w) 
        self.sess.run(self.z1.assign(self.f1_pred),
                      self.z2.assign(self.f2_pred),
                      self.z3.assign(self.f3_pred),
                      feed_dict={self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys})

        self.df = pd.DataFrame()

        self.run_NN()

    def initialize_variables(self):        
        # placeholders for training data
        self.x_data_tf = tf.placeholder(tf.float32, shape=[None, self.x_data.shape[1]])
        self.t_data_tf = tf.placeholder(tf.float32, shape=[None, self.t_data.shape[1]])
        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.E_tf = tf.placeholder(tf.float32, shape=[None, self.E.shape[1]])
        print('rhoshape: ' + str(self.rho.shape[1]))
        print('ushape: ' + str(self.u.shape[1]))
        print('Eshape: ' + str(self.E.shape[1]))
        print()
        # placeholders for training collocation points
        self.x_phys_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_phys_tf = tf.placeholder(tf.float32, shape=[None, 1])
        return

    def initialize_ADMM(self):
        # initialize the ADMM variables
        self.z1 = tf.Variable(tf.ones([self.N_f, 1]), dtype=tf.float32, trainable=False)
        self.z2 = tf.Variable(tf.ones([self.N_f, 1]), dtype=tf.float32, trainable=False)
        self.z3 = tf.Variable(tf.ones([self.N_f, 1]), dtype=tf.float32, trainable=False)
        self.lagrange1 = tf.Variable(tf.ones([self.N_f, 1]), dtype=tf.float32, trainable=False)
        self.lagrange2 = tf.Variable(tf.ones([self.N_f, 1]), dtype=tf.float32, trainable=False)
        self.lagrange3 = tf.Variable(tf.ones([self.N_f, 1]), dtype=tf.float32, trainable=False)
        self.pen = tf.constant(self.params.pen)
        self.recip_penNf = 1 / (self.pen * self.N_f)
        self.zeros = tf.zeros((self.N_f, 1))
        self.ones  = tf.ones((self.N_f, 1))

        # ADMM loss term for training the weights - use backprop on this
        self.loss = 1 / self.N_data * (tf.pow(tf.norm(self.rho_tf - self.rho_pred, 2), 2) + \
                    1 / self.N_data * tf.pow(tf.norm(self.u_tf - self.u_pred, 2), 2) + \
                    1 / self.N_data * tf.pow(tf.norm(self.E_tf - self.E_pred, 2), 2) + \
                    self.pen / 2 * tf.pow(tf.norm(self.f1_pred - self.z1 + self.lagrange1 / self.pen, 2), 2) + \
                    self.pen / 2 * tf.pow(tf.norm(self.f2_pred - self.z2 + self.lagrange2 / self.pen, 2), 2) + \
                    self.pen / 2 * tf.pow(tf.norm(self.f3_pred - self.z3 + self.lagrange3 / self.pen, 2), 2)
        
        self.lagrange1_update = self.lagrange1.assign(self.lagrange1 + self.pen * (self.f1_pred - self.z1))
        self.lagrange2_update = self.lagrange2.assign(self.lagrange2 + self.pen * (self.f2_pred - self.z2))
        self.lagrange3_update = self.lagrange3.assign(self.lagrange3 + self.pen * (self.f3_pred - self.z3))
        self.z1_update = self.z1.assign(self.compute_z(self.f1_pred, self.lagrange1))
        self.z2_update = self.z2.assign(self.compute_z(self.f2_pred, self.lagrange2))
        self.z3_update = self.z3.assign(self.compute_z(self.f3_pred, self.lagrange3))
        self.tol = 1e-4

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
            
    def net_rho_u_E(self, x, t):
        rho_u_E = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return rho_u_E
    
    def net_f(self, x, t):        
        rho_u_E = self.net_rho_u_E(x,t)  
        rho = rho_u_E[:,0:1]
        u = rho_u_E[:,1:2]
        E = rho_u_E[:,2:3]
        gamma = 1.4
        p = (gamma - 1)*(E - (1/2)*rho*(u**2))
        
        rho_t = tf.gradients(rho, t)[0]
        rhou_t = tf.gradients(rho*u, t)[0]
        E_t = tf.gradients(E, t)[0]
        
        rhou_x = tf.gradients(rho*u, x)[0]
        rhouu2_x = tf.gradients(rho*(u**2), x)[0] 
        p_x = tf.gradients(p, x)[0]
        uE_x = tf.gradients(u*E, x)[0]
        up_x = tf.gradients(u*p, x)[0]
             
        f1 = rho_t + rhou_x
        f2 = rhou_t + rhouu2_x + p_x
        f3 = E_t + uE_x + up_x
        
        return f1, f2, f3
    
    def callback(self, loss,):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss))
        
    def compute_z(self, f_pred, lagrange):
        val = f_pred + lagrange / self.pen
        # annoying digital logic workaround to implement conditional.
        # construct vectors of 1's and 0's that we can multiply
        # by the proper value and sum together
        cond1 = tf.where(tf.greater(val, self.recip_penNf), self.ones, self.zeros)
        cond3 = tf.where(tf.less(val, -1.0 * self.recip_penNf), self.ones, self.zeros)
        # cond2 is not needed since the complement of the intersection
        # of (cond1 and cond3) is cond2 and already assigned to 0

        dummy_z = cond1 * (val - self.recip_penNf) + cond3 * (val + self.recip_penNf)
        
        return dummy_z

    def train(self, nEpochs):
        tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, sef.rho_tf: self.rho, self.u_tf: self.u, self.E_tf: self.E,
                   self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}
        
        # training 
        start_time = time.time()
        epoch = 1
        num_Adam_iters = 1
            
        # train with physics
        while epoch < nEpochs:
            
            # perform the admm iteration
            for i in range(num_Adam_iters):
                self.sess.run(self.train_op_Adam, tf_dict)

            # new batch of collocation points
            self.x_phys = np.random.uniform(self.lb[0], self.ub[0], [self.params.N_f, 1])
            self.t_phys = np.random.uniform(self.lb[1], self.ub[1], [self.params.N_f, 1])
            tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.rho_tf: self.rho, self.u_tf: self.u, self.E_tf: self.E, 
                       self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}

            self.sess.run(self.z_update, tf_dict)
            self.sess.run(self.lagrange_update, tf_dict)
                    
            # print to monitor results
            if epoch % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss_value, elapsed))
                start_time = time.time()
                            
            # save figure every so often so if it crashes, we have some results
            if epoch % 10000 == 0:
                # self.plot_results()
                self.record_data(epoch)
                self.save_data()

            # increase the number of Adam training steps - cap at 20 for now
            if epoch % 1000 == 0 and num_Adam_iters <= 20:
                num_Adam_iters += 1
                
            epoch += 1

    def predict(self, X_star):
        
        tf_dict = {self.x_data_tf: X_star[:, 0:1], self.t_data_tf: X_star[:, 1:2],
                   self.x_phys_tf: X_star[:, 0:1], self.t_phys_tf: X_star[:, 1:2]}
        
        rho_pred_val = self.sess.run(self.rho_pred, tf_dict)
        u_pred_val = self.sess.run(self.u_pred, tf_dict)
        E_pred_val = self.sess.run(self.E_pred, tf_dict)
        f1_pred_val = self.sess.run(self.f1_pred, tf_dict)
        f2_pred_val = self.sess.run(self.f2_pred, tf_dict)
        f3_pred_val = self.sess.run(self.f3_pred, tf_dict)
        
        return rho_pred_val, u_pred_val, E_pred_val, f1_pred_val, f2_pred_val, f3_pred_val

    def load_data(self):
        # to make the filename string easier to read
        params = self.params
        self.filename = f'figures/ADMM/Abgrall_Euler/Staged_Deep/Nu{p.N_data}_Nf{p.N_f}_pen{int(p.pen)}_e{int(p.epochs)}.png'

        self.layers = [2, 200, 200, 200, 200, 200, 3]
        
        self.data = scipy.io.loadmat('../Data/Abgrall_eulers.mat')
        
        self.t = self.data['t'].flatten()[:, None]
        self.x = self.data['x'].flatten()[:, None]
        self.Exact_rho = np.real(self.data['rhosol']).T
        self.Exact_u = np.real(self.data['usol']).T
        self.Exact_E = np.real(self.data['Esol']).T
        
        self.X, self.T = np.meshgrid(self.x, self.t)
        
        self.X_star = np.hstack((self.X.flatten()[:, None], self.T.flatten()[:, None]))
        self.rho_star = self.Exact_rho.flatten()[:, None]
        self.u_star = self.Exact_u.flatten()[:, None]
        self.E_star = self.Exact_E.flatten()[:, None]
        
        # Domain bounds
        self.lb = self.X_star.min(0)
        self.ub = self.X_star.max(0)
        
        # Initial Condition
        domain_initial = np.hstack((self.X[0:1,:].T, self.T[0:1,:].T)) 
        initial_rho = self.Exact_rho[0:1,:].T 
        initial_u = self.Exact_u[0:1,:].T
        initial_E = self.Exact_E[0:1,:].T
        
        # Boundary Conditions
        domain_left_boundary = np.hstack((self.X[:,0:1], self.T[:,0:1])) 
        left_boundary_rho = self.Exact_rho[:,0:1] 
        left_boundary_u = self.Exact_u[:,0:1] 
        left_boundary_E = self.Exact_E[:,0:1] 
        domain_right_boundary = np.hstack((self.X[:,-1:], self.T[:,-1:])) 
        right_boundary_rho = self.Exact_rho[:,-1:] 
        right_boundary_u = self.Exact_u[:,-1:] 
        right_boundary_E = self.Exact_E[:,-1:] 
        
        self.X_data_train = np.vstack([domain_initial, domain_left_boundary, domain_right_boundary]) 
        self.rho_train = np.vstack([initial_rho, left_boundary_rho, right_boundary_rho]) 
        self.u_train = np.vstack([initial_u, left_boundary_u, right_boundary_u]) 
        self.E_train = np.vstack([initial_E, left_boundary_E, right_boundary_E]) 
                
        # Construct Training Data
        idx = np.random.choice(self.X_data_train.shape[0], self.params.N_data, replace=False) 
        self.X_data_train = self.X_data_train[idx, :]
        self.rho_train = self.rho_train[idx,:]
        self.u_train = self.u_train[idx,:]
        self.E_train = self.E_train[idx,:]
        
        # reassign here to conform to old workflow - NEEDS UPDATING
        self.x_data = self.X_data_train[:, 0:1]
        self.t_data = self.X_data_train[:, 1:2]
        self.rho = self.rho_train
        self.u = self.u_train
        self.E = self.E_train
             
    def run_NN(self):
        self.train(self.params.epochs)
        
        # calculate output statistics
        #self.plot_results()
        self.record_data(self.params.epochs)
        self.save_data()
        self.error_rho = np.linalg.norm(self.rho_star - self.rho_pred_val, 2) / np.linalg.norm(self.rho_star, 2)
        self.error_u = np.linalg.norm(self.u_star - self.u_pred_val, 2) / np.linalg.norm(self.u_star, 2)
        self.error_E = np.linalg.norm(self.E_star - self.E_pred_val, 2) / np.linalg.norm(self.E_star, 2)
        print('Error rho: %e %%' % (self.error_rho*100))
        print('Error u: %e %%' % (self.error_u*100))
        print('Error E: %e %%' % (self.error_E*100))
                
    def plot_results_u(self):
        print(self.filename)
        plt.rc('text', usetex=True)
        
        # calculate required statistics for plotting
        self.rho_pred_val, self.u_pred_val, self.E_pred_val, self.f1_pred_val, self.f2_pred_val, self.f3_pred_val = self.predict(self.X_star)
        self.u_pred_grid = griddata(self.X_star, self.u_pred_val.flatten(), (self.X, self.T), method='cubic')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        
        ####### Row 0: u(t,x) ##################
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1 - 0.06, bottom=1 - 1.0 / 3.0 + 0.06, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])
        
        h = ax.imshow(self.u_pred_grid.T, interpolation='nearest', cmap='rainbow',
                      extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        
        ax.plot(self.X_data_train[:, 1], self.X_data_train[:, 0], 'kx', label='Data (%d points)' % (self.u_train.shape[0]), markersize=2, clip_on=False)
        
        line = np.linspace(self.x.min(), self.x.max(), 2)[:, None]
        ax.plot(self.t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(self.t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(self.t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
        ax.set_title('$u(t,x)$', fontsize=18)
        
        ####### Row 1: u(t,x) slices ##################
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
        
        ax = plt.subplot(gs1[0, 0])
        ax.plot(self.x, self.Exact[25, :], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x, self.u_pred_grid[25, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 0.25$', fontsize=18)
        ax.axis('square')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        
        ax = plt.subplot(gs1[0, 1])
        ax.plot(self.x, self.Exact[50, :], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x, self.u_pred_grid[50, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.axis('square')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_title('$t = 0.50$', fontsize=18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
        
        ax = plt.subplot(gs1[0, 2])
        ax.plot(self.x, self.Exact[75, :], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x, self.u_pred_grid[75, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.axis('square')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_title('$t = 0.75$', fontsize=18)
                        
        plt.savefig(self.filename, dpi=300)
        plt.close()

        print()
        print('Figure saved to ' + self.filename)
        print()

        return

    def record_data(self, epoch_num):
        self.u_pred_val, self.f_pred_val = self.predict(self.X_star)
        self.rho_pred_val, self.u_pred_val, self.E_pred_val, self.f1_pred_val, self.f2_pred_val, self.f3_pred_val = self.predict(self.X_star)
        x = self.X_star[:, 0]
        t = self.X_star[:, 1]
        epoch = np.ones(len(x)) * epoch_num
        data = {'x': x, 't': t, 'rho_pred': self.rho_pred_val[:,0], 'u_pred': self.u_pred_val[:,0], 'E_pred': self.E_pred_val[:,0], 'epoch': epoch}
        self.df = pd.DataFrame(data)
        
    def save_data(self):
        self.df.to_csv(self.filename[:-3] + 'csv', mode='a', index=False)
        
    
if __name__ == "__main__":

    params = Parameters()
    if len(sys.argv) > 1:
        params.N_data = int(sys.argv[1])
        params.N_f = int(sys.argv[2])
        params.pen = float(sys.argv[3])
        params.epochs = int(sys.argv[4])
        params.gpu = str(sys.argv[5])
    A = PhysicsInformedNN(params)
    
