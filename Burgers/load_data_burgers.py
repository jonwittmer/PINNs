#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:07:14 2019

@author: hwan
"""

import numpy as np
import scipy.io

def load_data(run_options):
    if run_options.Burgers_Raissi == 1:
            training_data = scipy.io.loadmat('Data/burgers_shock.mat')
        
    if run_options.Burgers_Abgrall == 1:
            training_data = scipy.io.loadmat('Data/Abgrall_burgers_shock.mat')
        
    # Construct Domain
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
    idx = np.random.choice(X_u_train.shape[0], run_options.N_train, replace=False) 
    X_u_train = X_u_train[idx, :]
    x_data = X_u_train[:, 0:1]
    t_data = X_u_train[:, 1:2]
    u_train = u_train[idx,:]
    
    # randomly choose collocations points
    x_phys = np.random.uniform(lb[0], ub[0], [run_options.N_r, 1])
    t_phys = np.random.uniform(lb[1], ub[1], [run_options.N_r, 1])
                
    return Exact, x, t, X, T, X_star, lb, ub, u_star, X_u_train, x_data, t_data, u_train, x_phys, t_phys