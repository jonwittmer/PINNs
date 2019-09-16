#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:38:51 2019

@author: hwan
"""

import tensorflow as tf
import numpy as np
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

tf.set_random_seed(1234)

class PINN:
    def __init__(self, run_options, x_data_dimensions, t_data_dimensions, u_output_dimensions, lb, ub, nu, layers):
        
        self.layers = layers
        self.nu = nu
        
        # placeholders for training data
        self.x_data_tf = tf.placeholder(tf.float32, shape=[None, x_data_dimensions])
        self.t_data_tf = tf.placeholder(tf.float32, shape=[None, t_data_dimensions])
        self.u_train_tf = tf.placeholder(tf.float32, shape=[None, u_output_dimensions])
        
        # placeholders for training collocation points
        self.x_phys_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_phys_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # Initialize weights an bbiases
        self.weights = []
        self.biases = []
        num_layers = len(self.layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[self.layers[l], self.layers[l + 1]])
            b = tf.Variable(tf.zeros([1, self.layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            self.weights.append(W)
            self.biases.append(b)
        
        # Create data network and regularization network
        self.u_pred = self.net_u(self.x_data_tf, self.t_data_tf, lb, ub)
        self.r_pred = self.net_r(self.x_phys_tf, self.t_phys_tf, lb, ub)

    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, lb, ub):
        num_layers = len(self.weights) + 1        
        H = 2.0 * (X - lb) / (ub - lb) - 1.0
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        output = tf.add(tf.matmul(H, W), b)
        return output
    
    def net_u(self, x, t, lb, ub):
        u = self.neural_net(tf.concat([x, t], 1), lb, ub)
        return u
    
    def net_r(self, x, t, lb, ub):
        u = self.net_u(x, t, lb, ub)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        r = u_t +  u*u_x - self.nu * u_xx
        return r