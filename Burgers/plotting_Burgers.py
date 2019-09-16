#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:31:06 2019

@author: hwan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec

def plot_Burgers(run_options, u_pred, Exact, x, t, X, T, X_star, lb, ub, u_star, X_u_train, x_data, t_data, u_train):
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    
    plt.rc('text', usetex=True)

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
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize=18)  
    
    
    if run_options.Burgers_Raissi == 1:
        ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
        
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
        ax.set_title('$t = 0.75$', fontsize=18)\
        
    if run_options.Burgers_Abgrall == 1:
        ax.plot(t[0] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[128] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[-10] * np.ones((2, 1)), line, 'w-', linewidth=1)
        
        ####### Row 1: u(t,x) slices ##################   
        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
         
        ax = plt.subplot(gs1[0, 0])
        ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 0$', fontsize=18)
        ax.set_xlim([-0.1, np.pi + 0.1])
        ax.set_ylim([-0.1, 1.1])
        #ax.axis('square')
        
        ax = plt.subplot(gs1[0, 1])
        ax.plot(x, Exact[128, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x, U_pred[128, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        #ax.axis('square')
        ax.set_xlim([-0.1, np.pi + 0.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_title(f'$t = {np.round(t[128],2)}$', fontsize=18)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
        
        ax = plt.subplot(gs1[0, 2])
        ax.plot(x, Exact[-10, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x, U_pred[-10, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        #ax.axis('square')
        ax.set_xlim([-0.1, np.pi + 0.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_title(f'$t = {np.round(t[-10], 2)}$', fontsize=18)
    
    plt.savefig(run_options.figures_savefilepath, dpi=300)
    plt.close()

    print()
    print('Figure saved to ' + run_options.figures_savefilepath)
    print()    