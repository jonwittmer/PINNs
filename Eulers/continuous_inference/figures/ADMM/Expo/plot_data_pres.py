#!/usr/bin/env python3
# basic plotting file - eventually I want to make a video showing convergence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

# NN data
if len(sys.argv) < 2:
    raise Exception("The first argument must be the file name")

filename = sys.argv[1]
data = pd.read_csv(filename)

# check if custom index
if len(sys.argv) > 2:
    ind  = int(sys.argv[2])
else:
    ind = int(float(data.tail(1).epoch))

print("epochs: " + str(ind))
    
# original data from matlab solution: defines plotting grid
original_data = scipy.io.loadmat('../../../../Data/Abgrall_eulers.mat')
x = original_data['x'].flatten()[:, None]
t = original_data['t'].flatten()[:, None]
Exact_rho = np.real(original_data['rhosol']).T
Exact_u   = np.real(original_data['usol']).T
Exact_E   = np.real(original_data['Enersol']).T
X, T = np.meshgrid(x, t)

# select the relevant data
if not (data.x.dtype == np.float64 or data.x.dtype == np.int64):
    data = data[~(data.x.str.contains('x', na=False))]

# extract data from pandas dataframe
data.epoch = pd.to_numeric(data.epoch)
filtered_data = data[abs(data.epoch - ind) < 1]
X_star = filtered_data.x.to_numpy()
T_star = filtered_data.t.to_numpy()
rho_pred = filtered_data.rho_pred.to_numpy()
u_pred = filtered_data.u_pred.to_numpy()
E_pred = filtered_data.E_pred.to_numpy()
X_star = np.vstack((X_star, T_star))

Rho_pred = griddata(X_star.T, rho_pred.flatten(), (X, T), method='cubic')
U_pred = griddata(X_star.T, u_pred.flatten(), (X, T), method='cubic')
E_pred = griddata(X_star.T, E_pred.flatten(), (X, T), method='cubic')


plt.rc('text', usetex=True)
# plot for rho
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

midpoint = int(np.floor(len(Exact_rho[:,0])/2))

####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.1, bottom=1 - 1.0 / 2.0 + 0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(Rho_pred.T, interpolation='nearest', cmap='cividis',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[1] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[midpoint] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[-2] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
# ax.set_title('L1 Regularization with ADMM\n $u(t,x)$ - ' + str(ind) + ' epochs', fontsize=18)
ax.set_title('$L^1$ Regularization with ADMM\n $\\rho (t,x)$', fontsize=18)

####### Row 1: u(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
# gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
gs1.update(top=1 - 1.0 / 2.0 - 0.1, bottom=0.1, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_rho[1, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, Rho_pred[1, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (t,x)$')
ax.set_title('$t = 0$', fontsize=18)
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_rho[midpoint, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, Rho_pred[midpoint, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (t,x)$')
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[midpoint], 3)}$', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_rho[-2, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, Rho_pred[-2, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$\\rho (t,x)$')
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[-2], 3)}$', fontsize=18)


# plot for u
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.1, bottom=1 - 1.0 / 2.0 + 0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='cividis',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[1] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[midpoint] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[-2] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
# ax.set_title('L1 Regularization with ADMM\n $u(t,x)$ - ' + str(ind) + ' epochs', fontsize=18)
ax.set_title('$L^1$ Regularization with ADMM\n $u(t,x)$', fontsize=18)

####### Row 1: u(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
# gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
gs1.update(top=1 - 1.0 / 2.0 - 0.1, bottom=0.1, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_u[1, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[1, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0$', fontsize=18)
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_u[midpoint, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[midpoint, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[midpoint], 3)}$', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_u[-2, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[-2, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[-2][0], 3)}$', fontsize=18)




# plot for E
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.1, bottom=1 - 1.0 / 2.0 + 0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(E_pred.T, interpolation='nearest', cmap='cividis',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[1] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[midpoint] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[-2] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
# ax.set_title('L1 Regularization with ADMM\n $u(t,x)$ - ' + str(ind) + ' epochs', fontsize=18)
ax.set_title('$L^1$ Regularization with ADMM\n $E(t,x)$', fontsize=18)

####### Row 1: u(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
# gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
gs1.update(top=1 - 1.0 / 2.0 - 0.1, bottom=0.1, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_E[1, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, E_pred[1, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$E(t,x)$')
ax.set_title('$t = 0$', fontsize=18)
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_E[midpoint, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, E_pred[midpoint, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$E(t,x)$')
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[midpoint][0], 3)}$', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_E[-2, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, E_pred[-2, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$E(t,x)$')
#ax.set_xlim([-0.1, np.pi + 0.1])
#ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[-2][0], 3)}$', fontsize=18)


# plt.savefig(filename, dpi=300)
# plt.close()
plt.show()
