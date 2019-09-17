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
import time
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, '../../Utilities/')

from NN_PINNs_Burgers import PINN
from load_data_burgers import load_data
from numerical_integration import construct_trapezoidal_rule_scalar_multipliers
from plotting_Burgers import plot_Burgers

np.random.seed(1234)
tf.set_random_seed(1234)

###############################################################################
#                               Run Options                                   #
###############################################################################
class RunOptions:
    nu                = 0.0031831 # Burgers PDE parameter
    num_hidden_layers = 8
    num_hidden_nodes  = 20
    N_train           = 100
    N_r               = 50 # for l1 norm
    N_Int_x           = 10  # for L1 norm numerical integration
    N_Int_t           = 10  # for L1 norm numerical integration
    num_batch         = 100
    num_epochs        = 11
    gpu               = '3'

    # Choose PDE
    Burgers_Raissi = 0
    Burgers_Abgrall = 1
    
    # Choose Regularization
    PINNs_Regularization_l1 = 0
    PINNs_Regularization_Trapezoidal = 1
    
    # Setting Up File Names and Paths
    if Burgers_Raissi == 1:
        PDE = 'Raissi'
    
    if Burgers_Abgrall == 1:
        PDE = 'Abgrall'
        nu = 0 # No second derivative
    
    if PINNs_Regularization_l1 == 1:
        Regularization = 'l1'
        filename = 'Burgers_' + PDE + '_' + Regularization + '_hnodes%d_data%d_Nr%d_batch%d_epochs%d' %(num_hidden_nodes,N_train,N_r,num_batch,num_epochs)
    
    if PINNs_Regularization_Trapezoidal == 1:
        Regularization = 'Trape'
        filename = 'Burgers_' + PDE + '_' + Regularization + '_hnodes%d_data%d_Nx%d_Nt%d_batch%d_epochs%d' %(num_hidden_nodes,N_train,N_Int_x,N_Int_t,num_batch,num_epochs)
    
    figures_savefiledirectory = 'Figures/' + PDE + '/IRLS/' + Regularization + '/'
    outputs_savefiledirectory = 'Outputs/' + PDE + '/IRLS/' + Regularization + '/'
    
    figures_savefilepath = figures_savefiledirectory + filename
    outputs_savefilepath = outputs_savefiledirectory + filename
    
    # Creating Directories
    if not os.path.exists(figures_savefiledirectory):
        os.makedirs(figures_savefiledirectory)
        
    if not os.path.exists(outputs_savefiledirectory):
        os.makedirs(outputs_savefiledirectory)

###############################################################################
#                                  Functions                                  #
###############################################################################
def NN_u_star_predict(NN, X_star): # Note that X_star represents the full domain, not just the data points
    tf_dict = {NN.x_data_tf: X_star[:, 0:1], NN.t_data_tf: X_star[:, 1:2],
               NN.x_phys_tf: X_star[:, 0:1], NN.t_phys_tf: X_star[:, 1:2]} 
    u_star_pred = sess.run(NN.u_pred, tf_dict)  
    return u_star_pred

def save_prediction(savefilepath, epoch_num, u_pred):
    x = X_star[:, 0]
    t = X_star[:, 1]
    epoch = np.ones(len(x)) * epoch_num
    data = {'x': x, 't': t, 'u_pred': u_pred[:,0], 'epoch': epoch}
    df = pd.DataFrame(data)
    df.to_csv(savefilepath + '.csv', mode='a', index=False)
    
###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":
    
    run_options = RunOptions()
    
    ############################
    #   Load and Format Data   #
    ############################
    Exact, x, t, X, T, X_star, lb, ub, u_star, X_u_train, x_data, t_data, u_train, x_phys, t_phys = load_data(run_options)
    
    ###########################
    #   Training Properties   #
    ###########################   
    # Neural network
    NN = PINN(run_options, x_data.shape[1], t_data.shape[1], u_train.shape[1], lb, ub, run_options.nu)
    
    # Loss functional
    epsilon = 1e-15
    if run_options.PINNs_Regularization_l1 == 1:
        diag_entries = 1./(tf.math.sqrt(tf.math.abs(NN.r_pred + epsilon)))
        loss = 1 / run_options.N_train * tf.pow(tf.norm(u_train - NN.u_pred, 2), 2) + \
               1 / run_options.N_r * tf.pow(tf.norm(tf.diag(diag_entries)*NN.r_pred, 2), 2)
                    
    if run_options.PINNs_Regularization_Trapezoidal == 1:
        trapezoidal_scalars_x, trapezoidal_scalars_t, alpha, x_phys, t_phys = construct_trapezoidal_rule_scalar_multipliers(run_options.N_Int_x, run_options.N_Int_t, ub, lb)
        r_pred_trapezoidal = tf.multiply(trapezoidal_scalars_x, NN.r_pred)
        r_pred_trapezoidal = tf.multiply(trapezoidal_scalars_t, r_pred_trapezoidal)
        diag_entries = 1./(tf.math.sqrt(tf.math.abs(r_pred_trapezoidal + epsilon)))
        loss = 1/run_options.N_train * tf.pow(tf.norm(u_train - NN.u_pred, 2), 2) + \
                         alpha * tf.pow(tf.norm(tf.multiply(diag_entries,r_pred_trapezoidal), 2), 2)
                
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
                u_current_pred = NN_u_star_predict(NN, X_star)
                save_prediction(run_options.outputs_savefilepath, epoch, u_current_pred)         
                
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
        #lbfgs.minimize(sess, feed_dict=tf_dict)    
        
        ###############################
        #   Predictions and Plotting  #
        ################################          
        # Save final prediction
        u_final_pred = NN_u_star_predict(NN, X_star)
        save_prediction(run_options.outputs_savefilepath, run_options.num_epochs, u_final_pred)            
        
        # Plotting
        plot_Burgers(run_options, u_final_pred, Exact, x, t, X, T, X_star, lb, ub, u_star, X_u_train, x_data, t_data, u_train)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    