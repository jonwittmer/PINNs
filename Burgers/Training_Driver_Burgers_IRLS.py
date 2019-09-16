#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:01:06 2019

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

from NN_PINNs import PINNs

np.random.seed(1234)
tf.set_random_seed(1234)

###############################################################################
#                               Run Options                                   #
###############################################################################
class RunOptions:
    num_hidden_layers = 8
    num_hidden_nodes = 200
    N_train    = 100
    N_r    = 5000 # for l1 norm
    N_Int_x    = 100  # for L1 norm numerical integration
    N_Int_t    = 100  # for L1 norm numerical integration
    rho    = 40.0
    num_batch = 100
    num_epochs = 1e5
    gpu    = '3'

    # Choose PDE
    Burgers_Raissi = 1
    Burgers_Abgrall = 0
    
    # Choose Regularization
    PINNs_Regularization_l1 = 1
    PINNs_Regularization_Trapezoidal = 0
    
    # Setting Up File Names
    if Burgers_Raissi == 1:
        PDE = 'Raissi'
    
    if Burgers_Abgrall == 1:
        PDE = 'Abgrall'
    
    if PINNs_Regularization_l1 == 1:
        Regularization = 'l1'
        filename = 'Burgers_' + PDE + Regularization + '_hnodes%d_data%d_Nr%d_batch%d_epochs%d' %(num_hidden_nodes,N_train,N_r,num_batch,num_epochs)
    
    if PINNs_Regularization_Trapezoidal == 1:
        Regularization = 'Trape'
        filename = 'Burgers_' + PDE + Regularization + '_hnodes%d_data%d_Nx%d_Nt%d_batch%d_epochs%d' %(num_hidden_nodes,N_train,N_Int_x,N_Int_t,num_batch,num_epochs)
    
    figures_savefilepath = 'Figures/' + PDE + '/' + filename

###############################################################################
#                                  Functions                                  #
###############################################################################
def record_data(self, epoch_num):
    self.u_pred_val, self.f_pred_val = self.predict(self.X_star)
    x = self.X_star[:, 0]
    t = self.X_star[:, 1]
    epoch = np.ones(len(x)) * epoch_num
    data = {'x': x, 't': t, 'u_pred': self.u_pred_val[:,0], 'epoch': epoch}
    self.df = pd.DataFrame(data)
    
def save_data(self):
    self.df.to_csv(self.filename[:-3] + 'csv', mode='a', index=False)

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    run_options = RunOptions()
    
    #################
    #   Load Data   #
    #################
    training_data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t = training_data['t'].flatten()[:, None]
    x = training_data['x'].flatten()[:, None]
    Exact = np.real(training_data['usol']).T
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    
    # Domain Bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    # Initial Condition
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) 
    uu1 = Exact[0:1,:].T 
    
    xx2 = np.hstack((X[:,0:1], T[:,0:1])) 
    uu2 = Exact[:,0:1] 
    xx3 = np.hstack((X[:,-1:], T[:,-1:])) 
    uu3 = Exact[:,-1:] 
    
    # Full Set of Training Data
    X_u_train = np.vstack([xx1, xx2, xx3]) 
    u_train = np.vstack([uu1, uu2, uu3]) 
    
    # Construct Training Data Sample
    idx = np.random.choice(X_u_train.shape[0], run_options.N_data, replace=False) 
    X_u_train = X_u_train[idx, :]
    x_data = X_u_train[:, 0:1]
    t_data = X_u_train[:, 1:2]
    u_train = u_train[idx,:]
    
    # randomly choose collocations points
    x_phys = np.random.uniform(lb[0], ub[0], [run_options.N_r, 1])
    t_phys = np.random.uniform(lb[1], ub[1], [run_options.N_r, 1])
    
    ###########################
    #   Training Properties   #
    ###########################   
    layers = [2] + [run_options.num_hidden_nodes]*run_options.num_hidden_layers + [1]
    
    # Neural network
    NN = PINNs(run_options, X, layers)
    
    # Loss functional
    diag_entries = 1./(tf.math.sqrt(tf.math.abs(NN.f_pred)))
    loss = 1 / run_options.N_train * tf.pow(tf.norm(NN.u - NN.u_pred, 2), 2) + \
                1 / run_options.N_r * tf.pow(tf.norm(tf.diag(diag_entries)*NN.r_pred, 2), 2)
                
    # Set optimizers
    optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op_Adam = optimizer_Adam.minimize(loss)

    lbfgs = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                   method='L-BFGS-B',
                                                   options={'maxiter':1000,
                                                            'maxfun':50000,
                                                            'maxcor':50,
                                                            'maxls':50,
                                                            'ftol':1.0 * np.finfo(float).eps})
    
    # Set GPU configuration options
    gpu_options = tf.GPUOptions(visible_device_list= run_options.gpu,
                                allow_growth=True)
    
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=2,
                            gpu_options= gpu_options)
    
    ###########################
    #   Train Neural Network  #
    ###########################          
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables()) 
        
        # initial batch of collocation points
        tf_dict = {NN.x_data_tf: x_data, NN.t_data_tf: t_data, NN.u_train_tf: u_train,
                   NN.x_phys_tf: x_phys, NN.t_phys_tf: t_phys}
                
        # main iterations: updating Lagrange multiplier
        start_time = time.time()
        it = 0
        loss_value = 1000
        
        # store current weights to be updated later using IRLS
        weights_current = NN.weights

        for epoch in range(run_options.num_epochs): 
            sess.run(train_op_Adam, tf_dict)                    

            # print to monitor results
            if epoch % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = sess.run(loss, tf_dict)
                print(run_options.filename)
                print('GPU: ' + run_options.gpu)
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %(epoch, loss_value, elapsed))
                start_time = time.time()
                            
            # save figure every so often so if it crashes, we have some results
            if epoch % 10 == 0:
                #self.plot_results()
                record_data(it)
                save_data()
                
            # for iteratively reweighted least squares, the new weights are equal to the old weights plus the minimizer of the IRLS loss function    
            for l in range(0, len(NN.weights)): 
                NN.weights[l].assign(weights_current[l] + NN.weights[l]) 
                weights_current = NN.weights
            
            # new batch of collocation points
            x_phys = np.random.uniform(lb[0], ub[0], [run_options.N_r, 1])
            t_phys = np.random.uniform(lb[1], ub[1], [run_options.N_r, 1])
            tf_dict = {NN.x_data_tf: x_data, NN.t_data_tf: t_data, NN.u_train_tf: u_train,
                       NN.x_phys_tf: x_phys, NN.t_phys_tf: t_phys}           
        
        # Optimize with LBFGS
        print('Optimizing with LBFGS\n')        
        lbfgs.minimize(sess, feed_dict=tf_dict)    
        
        record_data(run_options.num_epochs)
        save_data()
        
###############################################################################
#                                  Plotting                                   #
###############################################################################
        print(run_options.filename)
        plt.rc('text', usetex=True)
        
        # Predictions       
        tf_dict = {NN.x_data_tf: X_star[:, 0:1], NN.t_data_tf: X_star[:, 1:2],
                   NN.x_phys_tf: X_star[:, 0:1], NN.t_phys_tf: X_star[:, 1:2]}        
        u_pred_val = sess.run(NN.u_pred, tf_dict)
        f_pred_val = sess.run(NN.f_pred, tf_dict)                
        
        U_pred = griddata(X_star, u_pred_val.flatten(), (X, T), method='cubic')
        
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
        ax.set_title('$u(t,x)$', fontsize=18)
        
        ####### Row 1: u(t,x) slices ##################
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
        
        ax = plt.subplot(gs1[0, 0])
        ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 0.25$', fontsize=18)
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
        ax.set_title('$t = 0.50$', fontsize=18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
        
        ax = plt.subplot(gs1[0, 2])
        ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.axis('square')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_title('$t = 0.75$', fontsize=18)
                        
        plt.savefig(run_options.filename, dpi=300)
        plt.close()

        print()
        print('Figure saved to ' + run_options.figures_savefilepath)
        print()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    