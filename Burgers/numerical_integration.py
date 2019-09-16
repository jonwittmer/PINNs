#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:53:54 2019

@author: hwan
"""

import numpy as np

def construct_trapezoidal_rule_scalar_multipliers(N_x, N_t, lb, ub):
    # construct scalar multipliers for trapezoidal rule
    x_int_points, spatial_step_size = np.linspace(lb[0],ub[0],N_x, retstep=True)
    t_int_points, time_step_size = np.linspace(lb[1],ub[1],N_t, retstep=True)
    X, T = np.meshgrid(x_int_points,t_int_points) # X is a (N_t x N_x) array with x_int_points repeated row wise N_t times. T is a (N_t x N_x) array with t_int_points repeated column wise N_x times
    x_t_int = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # Forms (N_xN_t x 2) array which associates each set of N_x spatial points (column 1) to one time point (column 2)
    x_phys = x_t_int[:, 0:1]
    t_phys = x_t_int[:, 1:2]
    trapezoidal_scalars_x = np.where(np.logical_or(abs(x_t_int[:,0] - lb[0]) < 1e-7, abs(x_t_int[:,0] - ub[0]) < 1e-7), np.ones(x_t_int.shape[0], dtype = np.float32), 2*np.ones(x_t_int.shape[0], dtype = np.float32))
    trapezoidal_scalars_t = np.where(np.logical_or(abs(x_t_int[:,1] - lb[1]) < 1e-7, abs(x_t_int[:,1] - ub[1]) < 1e-7), np.ones(x_t_int.shape[0], dtype = np.float32), 2*np.ones(x_t_int.shape[0], dtype = np.float32))  
    alpha = (spatial_step_size*time_step_size)/4    
    
    return trapezoidal_scalars_x, trapezoidal_scalars_t, alpha, x_phys, t_phys