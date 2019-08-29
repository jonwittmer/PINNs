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
    N_f    = 5000
    rho    = 40.0
    epochs = 1e5
    gpu    = '3'


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, params):
        
        self.params = params
        self.load_data()

        # Save dimensions
        self.N_u = self.params.N_u
        self.N_f = self.params.N_f
       
        # Initialize training variables
        self.weights, self.biases = self.initialize_NN(self.layers)
        
        self.initialize_variables()

        # evaluate outputs of network
        self.u_pred = self.net_u(self.x_data_tf, self.t_data_tf)
        self.f_pred = self.net_f(self.x_phys_tf, self.t_phys_tf)
        
        # construct loss function
        diag_entries = 1./(tf.math.sqrt(tf.math.abs(self.f_pred)))
        self.loss = 1 / self.N_u * tf.pow(tf.norm(self.u - self.u_pred, 2), 2) + \
                    1 / self.N_f * tf.pow(tf.norm(tf.diag(diag_entries)*self.f_pred, 2), 2)
        
        # set optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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
        

        
        # randomly choose collocations points
        self.x_phys = np.random.uniform(self.lb[0], self.ub[0], [self.params.N_f, 1])
        self.t_phys = np.random.uniform(self.lb[1], self.ub[1], [self.params.N_f, 1])

        self.df = pd.DataFrame()
        
        self.run_NN()

    def initialize_variables(self):
        # Initialize PDE parameters
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32, trainable=False)
        self.lambda_2 = tf.Variable([0.0031831], dtype=tf.float32, trainable=False)
        
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
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f
    
    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, lambda_2))
        
    def train(self, nIter):
        tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.u_tf: self.u,
                   self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}
        
        # main iterations: updating Lagrange multiplier
        start_time = time.time()
        it = 0
        loss_value = 1000

        while it < nIter:
            
            # perform the admm iteration
            self.sess.run(self.train_op_Adam, tf_dict)

            # new batch of collocation points
            self.x_phys = np.random.uniform(self.lb[0], self.ub[0], [self.params.N_f, 1])
            self.t_phys = np.random.uniform(self.lb[1], self.ub[1], [self.params.N_f, 1])
            tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.u_tf: self.u,
                       self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}
                    
            # print to monitor results
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()
                            
            # save figure every so often so if it crashes, we have some results
            if it % 10000 == 0:
                #self.plot_results()
                self.record_data(it)
                self.save_data()
                
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
        self.filename = f'figures/L2/Nu{p.N_u}_Nf{p.N_f}_e{int(p.epochs)}.png'

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
        

        ##############################
        #   Construct Training Data  #
        ##############################
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
    
