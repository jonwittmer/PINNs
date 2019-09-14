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
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')


np.random.seed(1234)
tf.set_random_seed(1234)


class Parameters:
    N_u    = 100
    N_x    = 20
    N_t    = 20
    epochs = 10
    gpu    = '3'


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, params):
        
        self.params = params
        self.load_data()
        self.nu = 0.0031831 # parameter for PDE

        # Save dimensions
        self.N_u = self.params.N_u
        self.N_x = self.params.N_x
        self.N_t = self.params.N_t
       
        # Initialize training variables
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.initialize_placeholders()

        # evaluate outputs of network
        self.u_pred = self.net_u(self.x_data_tf, self.t_data_tf)
        self.f_pred = self.net_f(self.x_phys_tf, self.t_phys_tf)
        
        # construct L1-regularization term with trapezoidal rule
        spatial_step_size = (self.ub[0]-self.lb[0])/self.N_x
        x_int_points = np.arange(self.lb[0],self.ub[0],spatial_step_size)
        time_step_size = (self.ub[1]-self.lb[1])/self.N_t
        t_int_points = np.arange(self.lb[1],self.ub[1],time_step_size)
        X, T = np.meshgrid(x_int_points,t_int_points) # X is a (N_t x N_x) array with x_int_points repeated row wise N_t times. T is a (N_t x N_x) array with t_int_points repeated column wise N_x times
        x_t_int = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # Forms (N_xN_t x 2) array which associates each set of N_x spatial points (column 1) to one time point (column 2)
        self.x_phys = x_t_int[:, 0:1]
        self.t_phys = x_t_int[:, 1:2]
        trapezoidal_scalars_x = tf.where(np.logical_or(x_t_int[:,0] == self.lb[0], x_t_int[:,0] == self.ub[0]), np.ones(x_t_int.shape[0], dtype = np.float32), 2*np.ones(x_t_int.shape[0], dtype = np.float32))
        trapezoidal_scalars_t = tf.where(np.logical_or(x_t_int[:,1] == self.lb[1], x_t_int[:,1] == self.ub[1]), np.ones(x_t_int.shape[0], dtype = np.float32), 2*np.ones(x_t_int.shape[0], dtype = np.float32))  
        self.f_pred_trapezoidal = tf.multiply(trapezoidal_scalars_x,self.f_pred)
        self.f_pred_trapezoidal = tf.multiply(trapezoidal_scalars_t,self.f_pred_trapezoidal)
        self.alpha = (spatial_step_size*time_step_size)/4
        
        # construct loss function
        self.diag_entries = 1./(tf.math.sqrt(tf.math.abs(self.f_pred_trapezoidal)))
        self.loss_IRLS = 1/self.N_u * tf.pow(tf.norm(self.u - self.u_pred, 2), 2) + \
                         self.alpha * tf.pow(tf.norm(tf.multiply(self.diag_entries,self.f_pred_trapezoidal), 2), 2)
        
        # set optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss_IRLS)

        # set l-bfgs optimizer
        self.lbfgs = tf.contrib.opt.ScipyOptimizerInterface(self.loss_IRLS,
                                                             method='L-BFGS-B',
                                                             options={'maxiter':1000,
                                                                      'maxfun':50000,
                                                                      'maxcor':50,
                                                                      'maxls':50,
                                                                      'ftol':1.0 * np.finfo(float).eps})
        
        self.weights_current = self.weights
        self.weights_update_flag = tf.Variable(0)
        self.update_weights = self.weights_update_flag.assign(self.compute_weights())
        
        # set configuration options
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
        
        self.df = pd.DataFrame()
        
        self.run_NN()

    def initialize_placeholders(self):        
        # placeholders for training data
        self.x_data_tf = tf.placeholder(tf.float32, shape=[None, self.x_data.shape[1]])
        self.t_data_tf = tf.placeholder(tf.float32, shape=[None, self.t_data.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        print('ushape: ' + str(self.u.shape[1]))
        print()
        # placeholders for training collocation points
        self.x_phys_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_phys_tf = tf.placeholder(tf.float32, shape=[None, 1])
        return

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
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t +  u*u_x - self.nu * u_xx
        return f
    
    def compute_weights(self):
        for l in range(0, len(self.weights)): 
           self.weights[l].assign(self.weights_current[l] + self.weights[l]) 
        
        self.weights_current = self.weights
        return 1

    def train(self, nIter):
        # initial batch of collocation points
        tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.u_tf: self.u,
                   self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}
        
        # main iterations: updating Lagrange multiplier
        start_time = time.time()
        it = 0
        loss_value = 1000
        
        # store current weights to be updated later using IRLS
        self.weights_current = self.weights

        while it < nIter:
            # perform optimization
            #self.sess.run(self.train_op_Adam, tf_dict)                    
            self.lbfgs.minimize(self.sess, feed_dict=tf_dict)

            # print to monitor results
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_IRLS, tf_dict)
                print(self.filename[:-3])
                print('GPU: ' + self.params.gpu)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()
                            
            # save figure every so often so if it crashes, we have some results
            if it % 10 == 0:
                #self.plot_results()
                self.record_data(it)
                self.save_data()
                
            # for iteratively reweighted least squares, the new weights are equal to the old weights plus the minimizer of the IRLS loss function    
            self.sess.run(self.update_weights, feed_dict=tf_dict)
                        
            it += 1

    def predict(self, X_star):        
        tf_dict = {self.x_data_tf: X_star[:, 0:1], self.t_data_tf: X_star[:, 1:2],
                   self.x_phys_tf: X_star[:, 0:1], self.t_phys_tf: X_star[:, 1:2]}        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)        
        return u_star, f_star

    def load_data(self):
        # to make the filename string easier to read
        p = self.params
        self.filename = f'figures/IRLS/Raissi_PDE/Trape_IRLS_Nu{p.N_u}_Nx{p.N_x}_Nt{p.N_t}_e{int(p.epochs)}.png'

        self.layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        
        self.data = scipy.io.loadmat('../Data/burgers_shock.mat')
        
        self.t = self.data['t'].flatten()[:, None]
        self.x = self.data['x'].flatten()[:, None]
        self.Exact = np.real(self.data['usol']).T
        
        self.X, self.T = np.meshgrid(self.x, self.t)
        
        self.X_star = np.hstack((self.X.flatten()[:, None], self.T.flatten()[:, None]))
        self.u_star = self.Exact.flatten()[:, None]
        
        # Domain bounds
        self.lb = self.X_star.min(0)
        self.ub = self.X_star.max(0)
        
        # Initial Condition
        xx1 = np.hstack((self.X[0:1,:].T, self.T[0:1,:].T)) 
        uu1 = self.Exact[0:1,:].T 
        
        xx2 = np.hstack((self.X[:,0:1], self.T[:,0:1])) 
        uu2 = self.Exact[:,0:1] 
        xx3 = np.hstack((self.X[:,-1:], self.T[:,-1:])) 
        uu3 = self.Exact[:,-1:] 
        
        self.X_u_train = np.vstack([xx1, xx2, xx3]) 
        self.u_train = np.vstack([uu1, uu2, uu3]) 
        
        # Construct Training Data
        idx = np.random.choice(self.X_u_train.shape[0], self.params.N_u, replace=False) 
        self.X_u_train = self.X_u_train[idx, :]
        self.u_train = self.u_train[idx,:]
        
        # reassign here to conform to old workflow - NEEDS UPDATING
        self.x_data = self.X_u_train[:, 0:1]
        self.t_data = self.X_u_train[:, 1:2]
        self.u = self.u_train
             
    def run_NN(self):
        self.train(self.params.epochs)
        
        # calculate output statistics
        #self.plot_results()
        self.record_data(self.params.epochs)
        self.save_data()
        self.error_u = np.linalg.norm(self.u_star - self.u_pred_val, 2) / np.linalg.norm(self.u_star, 2)
        print('Error u: %e %%' % (self.error_u*100))
                
    def plot_results(self):
        print(self.filename)
        plt.rc('text', usetex=True)
        
        # calculate required statistics for plotting
        self.u_pred_val, self.f_pred_val = self.predict(self.X_star)
        self.U_pred = griddata(self.X_star, self.u_pred_val.flatten(), (self.X, self.T), method='cubic')
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        
        ####### Row 0: u(t,x) ##################
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1 - 0.06, bottom=1 - 1.0 / 3.0 + 0.06, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])
        
        h = ax.imshow(self.U_pred.T, interpolation='nearest', cmap='rainbow',
                      extent=[self.t.min(), self.t.max(), self.x.min(), self.x.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        
        ax.plot(self.X_u_train[:, 1], self.X_u_train[:, 0], 'kx', label='Data (%d points)' % (self.u_train.shape[0]), markersize=2, clip_on=False)
        
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
        ax.plot(self.x, self.U_pred[25, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 0.25$', fontsize=18)
        ax.axis('square')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        
        ax = plt.subplot(gs1[0, 1])
        ax.plot(self.x, self.Exact[50, :], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x, self.U_pred[50, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.axis('square')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_title('$t = 0.50$', fontsize=18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
        
        ax = plt.subplot(gs1[0, 2])
        ax.plot(self.x, self.Exact[75, :], 'b-', linewidth=2, label='Exact')
        ax.plot(self.x, self.U_pred[75, :], 'r--', linewidth=2, label='Prediction')
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
        x = self.X_star[:, 0]
        t = self.X_star[:, 1]
        epoch = np.ones(len(x)) * epoch_num
        data = {'x': x, 't': t, 'u_pred': self.u_pred_val[:,0], 'epoch': epoch}
        self.df = pd.DataFrame(data)
        
    def save_data(self):
        self.df.to_csv(self.filename[:-3] + 'csv', mode='a', index=False)
        
    
if __name__ == "__main__":
     p = Parameters()
     A = PhysicsInformedNN(p)
    
