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
    N_data = 150
    N_f    = 20000
    pen    = 10.0
    epochs = 1e6
    gpu    = '2'


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, params):
        
        self.params = params
        self.load_data()
        
        # Save dimensions
        self.N_data = self.params.N_data
        self.N_f = self.params.N_f
       
        # Initialize training variables
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.initialize_placeholders()

        # Evaluate outputs of network
        rho_u_E_pred = self.net_rho_u_E(self.x_data_tf, self.t_data_tf)
        self.rho_pred = rho_u_E_pred[:,0:1]
        self.u_pred = rho_u_E_pred[:,1:2]
        self.E_pred = rho_u_E_pred[:,2:3]
        self.f1_pred, self.f2_pred, self.f3_pred = self.net_f(self.x_phys_tf, self.t_phys_tf)
        
        self.loss = 1 / self.N_data * tf.pow(tf.norm(self.rho_tf - self.rho_pred, 2), 2) + \
                    1 / self.N_data * tf.pow(tf.norm(self.u_tf - self.u_pred,     2), 2) + \
                    1 / self.N_data * tf.pow(tf.norm(self.E_tf - self.E_pred,     2), 2) + \
                    1 / self.N_f * tf.pow(tf.norm(self.f1_pred, 2), 2) + \
                    1 / self.N_f * tf.pow(tf.norm(self.f2_pred, 2), 2) + \
                    1 / self.N_f * tf.pow(tf.norm(self.f2_pred, 2), 2)

        # Optimizer
        self.optimizer_Adam  = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam   = self.optimizer_Adam.minimize(self.loss)
        
        # 2nd order optimizer used once we get "close" to the solution
        self.tol = tf.placeholder(tf.float64, shape=[])
        self.lbfgs = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                           method='L-BFGS-B',
                                                           options={'maxiter': 10000,
                                                                    'maxfun': 50000,
                                                                    'maxcor': 50,
                                                                    'maxls': 50,
                                                                    'ftol':1e-12})
                                                                    #'ftol':1e6 * np.finfo(float).eps})
                                                                    #'ftol':self.tol})
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
        self.t_phys = self.lb[1] + self.exponential_time_sample(np.zeros((self.params.N_f, 1)), self.ub[1] - self.lb[1])
        #self.t_phys = np.random.uniform(self.lb[1], self.ub[1], [self.params.N_f,1])

        self.df = pd.DataFrame()

        self.run_NN()

    def initialize_placeholders(self):        
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

    def exponential_time_sample(self, empty_array, ub):
        # exponential distribution where CDF = 0.9 at upper bound
        # truncate this distribution so that we stil have sufficient points
        # near the upper bound of the time domain
        beta = ub / np.log(10)
        data = np.random.exponential(beta, size=(empty_array.shape[0]*2, 1))
        
        # remove data that is 
        data = data[(data[:,0] < ub),:]
        
        # make sure we have enough data before returninng
        if data.shape[0] < empty_array.shape[0]:
            data = self.exponential_time_sample(empty_array, ub)
        
        return data[0:empty_array.shape[0], 0:1]            

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
        
    def train(self, nEpochs):
        self.current_tol = 1e-7
        tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.rho_tf: self.rho, self.u_tf: self.u, self.E_tf: self.E,
                   self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys, self.tol: self.current_tol}
        
        # training 
        start_time = time.time()
        it = 1
                   
        # train with physics
        while it < nEpochs:
            tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.rho_tf: self.rho, self.u_tf: self.u, self.E_tf: self.E,
                       self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys, self.tol: self.current_tol}
            
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # new batch of collocation points
            #self.x_phys = np.random.uniform(self.lb[0], self.ub[0], [self.params.N_f, 1])
            #self.t_phys = np.random.uniform(self.lb[1], self.ub[1], [self.params.N_f, 1])
            #tf_dict = {self.x_data_tf: self.x_data, self.t_data_tf: self.t_data, self.rho_tf: self.rho, self.u_tf: self.u, self.E_tf: self.E, 
            #           self.x_phys_tf: self.x_phys, self.t_phys_tf: self.t_phys}

            # print to monitor results
            if it <= 1e7:
                if it % 1000 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' %
                          (it, loss_value, elapsed))
                    start_time = time.time()
                            
                # save figure every so often so if it crashes, we have some results
                if it % 10000 == 0:
                    self.record_data(it)
                    self.save_data()
            else:
                if it % 1 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' %
                          (it, loss_value, elapsed))
                    start_time = time.time()
                            
                # save figure every so often so if it crashes, we have some results
                if it % 100 == 0:
                    self.record_data(it)
                    self.save_data()

            it += 1

    def predict(self, X_star):
        
        tf_dict = {self.x_data_tf: X_star[:, 0:1], self.t_data_tf: X_star[:, 1:2],
                   self.x_phys_tf: X_star[:, 0:1], self.t_phys_tf: X_star[:, 1:2]}
        
        rho_pred_val = self.sess.run(self.rho_pred, tf_dict)
        u_pred_val   = self.sess.run(self.u_pred, tf_dict)
        E_pred_val   = self.sess.run(self.E_pred, tf_dict)
        f1_pred_val  = self.sess.run(self.f1_pred, tf_dict)
        f2_pred_val  = self.sess.run(self.f2_pred, tf_dict)
        f3_pred_val  = self.sess.run(self.f3_pred, tf_dict)
        
        return rho_pred_val, u_pred_val, E_pred_val, f1_pred_val, f2_pred_val, f3_pred_val

    def load_data(self):
        params = self.params
        
        n0 = 50 # Number of data points for initial condition
        nb = 50 # Number of data points for boundary condition

        self.layers = [2, 200, 200, 200, 200, 200, 3]
        
        self.data = scipy.io.loadmat('../Data/Abgrall_eulers.mat')
        
        self.t = self.data['t'].flatten()[:, None]
        self.x = self.data['x'].flatten()[:, None]
        self.Exact_rho = np.real(self.data['rhosol']).T
        self.Exact_u = np.real(self.data['usol']).T
        self.Exact_E = np.real(self.data['Enersol']).T
        
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
        
        idx_x = np.random.choice(self.x.shape[0], n0, replace=False) #Extract Data Points
        domain_initial = domain_initial[idx_x,:]
        initial_rho = initial_rho[idx_x,0:1]
        initial_u = initial_u[idx_x,0:1]
        initial_E = initial_E[idx_x,0:1]
        
        # Boundary Conditions
        domain_left_boundary = np.hstack((self.X[:,0:1], self.T[:,0:1]))
        left_boundary_rho = self.Exact_rho[:,0:1] 
        left_boundary_u = self.Exact_u[:,0:1] 
        left_boundary_E = self.Exact_E[:,0:1] 
        domain_right_boundary = np.hstack((self.X[:,-1:], self.T[:,-1:])) 
        right_boundary_rho = self.Exact_rho[:,-1:] 
        right_boundary_u = self.Exact_u[:,-1:] 
        right_boundary_E = self.Exact_E[:,-1:] 
        
        idx_t = np.random.choice(self.t.shape[0], nb, replace=False) #Extract Data Points
        domain_left_boundary = domain_left_boundary[idx_t,:]
        left_boundary_rho = left_boundary_rho[idx_t,0:1]
        left_boundary_u = left_boundary_u[idx_t,0:1]
        left_boundary_E = left_boundary_E[idx_t,0:1]
        domain_right_boundary = domain_right_boundary[idx_t,:]
        right_boundary_rho = right_boundary_rho[idx_t,0:1]
        right_boundary_u = right_boundary_u[idx_t,0:1]
        right_boundary_E = right_boundary_E[idx_t,0:1]
        
        self.X_data_train = np.vstack([domain_initial, domain_left_boundary, domain_right_boundary]) 
        self.rho_train = np.vstack([initial_rho, left_boundary_rho, right_boundary_rho]) 
        self.u_train = np.vstack([initial_u, left_boundary_u, right_boundary_u]) 
        self.E_train = np.vstack([initial_E, left_boundary_E, right_boundary_E]) 
        
        # use all of the input data
        '''
        # Construct Training Data
        idx = np.random.choice(self.X_data_train.shape[0], self.params.N_data, replace=False) 
        self.X_data_train = self.X_data_train[idx, :]
        self.rho_train = self.rho_train[idx,:]
        self.u_train = self.u_train[idx,:]
        self.E_train = self.E_train[idx,:]
        '''
        self.N_data = len(self.X_data_train[:,0])

        # reassign here to conform to old workflow - NEEDS UPDATING
        self.x_data = self.X_data_train[:, 0:1]
        self.t_data = self.X_data_train[:, 1:2]
        self.rho = self.rho_train
        self.u = self.u_train
        self.E = self.E_train

        # to make the filename string easier to read
        self.filename = f'figures/L2/Expo/Nu{self.N_data}_Nf{params.N_f}_pen{int(params.pen)}_e{int(params.epochs)}.png'
             
    def run_NN(self):
        self.train(self.params.epochs)
        
        # calculate output statistics
        self.record_data(self.params.epochs)
        self.save_data()
        self.error_rho = np.linalg.norm(self.rho_star - self.rho_pred_val, 2) / np.linalg.norm(self.rho_star, 2)
        self.error_u = np.linalg.norm(self.u_star - self.u_pred_val, 2) / np.linalg.norm(self.u_star, 2)
        self.error_E = np.linalg.norm(self.E_star - self.E_pred_val, 2) / np.linalg.norm(self.E_star, 2)
        print('Error rho: %e %%' % (self.error_rho*100))
        print('Error u: %e %%' % (self.error_u*100))
        print('Error E: %e %%' % (self.error_E*100))
                
    def record_data(self, epoch_num):
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
        params.epochs = int(sys.argv[3])
        params.gpu = str(sys.argv[4])
    A = PhysicsInformedNN(params)
    
