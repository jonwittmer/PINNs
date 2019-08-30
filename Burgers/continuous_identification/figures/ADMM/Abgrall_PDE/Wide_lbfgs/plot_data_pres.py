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

# original data from matlab solution: defines plotting grid
original_data = scipy.io.loadmat('../../../../../Data/Abgrall_burgers_shock.mat')
x = original_data['x'].flatten()[:, None]
t = original_data['t'].flatten()[:, None]
Exact = np.real(original_data['usol']).T
X, T = np.meshgrid(x, t)

# select the relevant data
if not (data.x.dtype == np.float64 or data.x.dtype == np.int64):
    data = data[~(data.x.str.contains('x', na=False))]


data.epoch = pd.to_numeric(data.epoch)
filtered_data = data[abs(data.epoch - ind) < 1]
X_star = filtered_data.x.to_numpy()
T_star = filtered_data.t.to_numpy()
u_pred = filtered_data.u_pred.to_numpy()
X_star = np.vstack((X_star, T_star))

U_pred = griddata(X_star.T, u_pred.flatten(), (X, T), method='cubic')

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.1, bottom=1 - 1.0 / 2.0 + 0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[0] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[128] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[-10] * np.ones((2, 1)), line, 'w-', linewidth=1)

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
ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0$', fontsize=18)
ax.set_xlim([-0.1, np.pi + 0.1])
ax.set_ylim([-0.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[128, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[128, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_xlim([-0.1, np.pi + 0.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[128],2)}$', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[-10, :], 'b-', linewidth=2, label='Exact')
ax.plot(x, U_pred[-10, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_xlim([-0.1, np.pi + 0.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[-10], 2)}$', fontsize=18)

# plt.savefig(filename, dpi=300)
# plt.close()
plt.show()
