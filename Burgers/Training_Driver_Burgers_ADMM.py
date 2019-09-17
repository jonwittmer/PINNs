#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:10:17 2019

@author: hwan
"""

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
    N_r               = 5000 # for l1 norm
    N_Int_x           = 100  # for L1 norm numerical integration
    N_Int_t           = 100  # for L1 norm numerical integration
    pen               = 10.0 # Penalty parameter for augmented Lagrangian, needs to be float32
    num_epochs        = 50000
    gpu               = '1'

    # Choose PDE
    Burgers_Raissi = 1
    Burgers_Abgrall = 0
    
    # Choose Regularization
    PINNs_Regularization_l1 = 1
    PINNs_Regularization_Trapezoidal = 0 # Not yet coded
    
    # Setting Up File Names and Paths
    if Burgers_Raissi == 1:
        PDE = 'Raissi'
    
    if Burgers_Abgrall == 1:
        PDE = 'Abgrall'
        nu = 0 # No second derivative
    
    if PINNs_Regularization_l1 == 1:
        Regularization = 'l1'
        filename = 'Burgers_' + PDE + '_' + Regularization + '_hnodes%d_data%d_Nr%d_epochs%d' %(num_hidden_nodes, N_train, N_r, num_epochs)
    
    if PINNs_Regularization_Trapezoidal == 1:
        N_r = N_Int_x*N_Int_t
        Regularization = 'Trape'
        filename = 'Burgers_' + PDE + '_' + Regularization + '_hnodes%d_data%d_Nx%d_Nt%d_epochs%d' %(num_hidden_nodes, N_train, N_Int_x, N_Int_t, num_epochs)
    
    figures_savefiledirectory = 'Figures/' + PDE + '/ADMM/' + Regularization + '/'
    outputs_savefiledirectory = 'Outputs/' + PDE + '/ADMM/' + Regularization + '/'
    
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
    
def compute_z(r_pred, alpha, pen):
    val = r_pred + lagrange / pen

    # annoying digital logic workaround to implement conditional.
    # construct vectors of 1's and 0's that we can multiply
    # by the proper value and sum together
    cond1 = tf.where(tf.greater(val, alpha/pen), ones, zeros)
    cond3 = tf.where(tf.less(val, - 1.0 * alpha/pen), ones, zeros)
    # cond2 is not needed since the complement of the intersection
    # of (cond1 and cond3) is cond2 and already assigned to 0

    dummy_z = cond1 * (val - alpha/pen) + cond3 * (val + alpha/pen)
    
    return dummy_z
        
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
    
    # Initialize ADMM objects
    z = tf.Variable(tf.ones([run_options.N_r, 1]), dtype=tf.float32, trainable=False)
    lagrange = tf.Variable(tf.ones([run_options.N_r, 1]), dtype=tf.float32, trainable=False)
    pen = tf.constant(run_options.pen)
    alpha = 1/run_options.N_r # replaced below for trapezoidal rule
    zeros = tf.zeros((run_options.N_r, 1))
    ones  = tf.ones((run_options.N_r, 1))   
    lagrange_update = lagrange.assign(lagrange + pen * (NN.r_pred - z))
    z_update = z.assign(compute_z(NN.r_pred, alpha, pen))
    
    # Loss functional
    epsilon = 1e-15
    if run_options.PINNs_Regularization_l1 == 1:
        loss = 1/run_options.N_train * tf.pow(tf.norm(u_train - NN.u_pred, 2), 2) + \
                               pen/2 * tf.pow(tf.norm(NN.r_pred - z + lagrange/pen, 2), 2)
                    
    if run_options.PINNs_Regularization_Trapezoidal == 1:
        trapezoidal_scalars_x, trapezoidal_scalars_t, alpha, x_phys, t_phys = construct_trapezoidal_rule_scalar_multipliers(run_options.N_Int_x, run_options.N_Int_t, ub, lb)
        r_pred_trapezoidal = tf.multiply(trapezoidal_scalars_x, NN.r_pred)
        r_pred_trapezoidal = tf.multiply(trapezoidal_scalars_t, r_pred_trapezoidal)        
        loss = 1/run_options.N_train * tf.pow(tf.norm(u_train - NN.u_pred, 2), 2) + \
                               pen/2 * tf.pow(tf.norm(r_pred_trapezoidal - z + lagrange/pen, 2), 2)
                
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
        
        # assign the real initial value of z = r(w) 
        sess.run(z.assign(NN.r_pred), feed_dict={NN.x_phys_tf: x_phys, NN.t_phys_tf: t_phys})
        
        # initial batch of collocation points
        tf_dict = {NN.x_data_tf: x_data, NN.t_data_tf: t_data, NN.u_train_tf: u_train,
                   NN.x_phys_tf: x_phys, NN.t_phys_tf: t_phys}
                
        # main iterations: updating Lagrange multiplier
        start_time = time.time()
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
             
            sess.run(z_update, tf_dict)
            sess.run(lagrange_update, tf_dict)   
                
            # save figure every so often so if it crashes, we have some results
            if epoch % 10 == 0:
                u_current_pred = NN_u_star_predict(NN, X_star)
                save_prediction(run_options.outputs_savefilepath, epoch, u_current_pred)  
            
            # new batch of collocation points
            x_phys = np.random.uniform(lb[0], ub[0], [run_options.N_r, 1])
            t_phys = np.random.uniform(lb[1], ub[1], [run_options.N_r, 1])
            tf_dict = {NN.x_data_tf: x_data, NN.t_data_tf: t_data, NN.u_train_tf: u_train,
                       NN.x_phys_tf: x_phys, NN.t_phys_tf: t_phys}           
        
        # Optimize with LBFGS
        print('Optimizing with LBFGS\n')        
        lbfgs.minimize(sess, feed_dict=tf_dict)    
        
        ###############################
        #   Predictions and Plotting  #
        ################################    
        # Save final prediction
        u_final_pred = NN_u_star_predict(NN, X_star)
        save_prediction(run_options.outputs_savefilepath, run_options.num_epochs, u_final_pred) 
        print('Final Prediction Saved\n') 
        
        # Plotting
        plot_Burgers(run_options, u_final_pred, Exact, x, t, X, T, X_star, lb, ub, u_star, X_u_train, x_data, t_data, u_train)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    