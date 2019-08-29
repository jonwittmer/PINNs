#!/usr/bin/env python3
# basic plotting file - eventually I want to make a video showing convergence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
all_epochs = data.epoch.unique()

filtered_data = data[abs(data.epoch - all_epochs[0]) < 1]
X_star = filtered_data.x.to_numpy()
T_star = filtered_data.t.to_numpy()
u_pred = filtered_data.u_pred.to_numpy()
X_star = np.vstack((X_star, T_star))

U_pred = griddata(X_star.T, u_pred.flatten(), (X, T), method='cubic')

fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

####### Row 0: u(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1 - 0.06, bottom=1 - 1.0 / 3.0 + 0.06, left=0.15, right=0.85, wspace=0)
ax_top = plt.subplot(gs0[:, :])

h = ax_top.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax_top)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax_top.plot(t[0] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax_top.plot(t[128] * np.ones((2, 1)), line, 'w-', linewidth=1)
ax_top.plot(t[-10] * np.ones((2, 1)), line, 'w-', linewidth=1)

ax_top.set_xlabel('$t$')
ax_top.set_ylabel('$x$')
ax_top.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
ax_top.set_title('$u(t,x)$ - ' + str(ind) + ' epochs', fontsize=18)

####### Row 1: u(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact[0, :], 'b-', linewidth=2, label='Exact')
first_line, = ax.plot(x, U_pred[0, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0$', fontsize=18)
ax.set_xlim([-0.1, np.pi + 0.1])
ax.set_ylim([-0.1, 1.1])
#ax.axis('square')

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[128, :], 'b-', linewidth=2, label='Exact')
second_line, = ax.plot(x, U_pred[128, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
#ax.axis('square')
ax.set_xlim([-0.1, np.pi + 0.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[128],2)}$', fontsize=18)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact[-10, :], 'b-', linewidth=2, label='Exact')
third_line, = ax.plot(x, U_pred[-10, :], 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
#ax.axis('square')
ax.set_xlim([-0.1, np.pi + 0.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title(f'$t = {np.round(t[-10], 2)}$', fontsize=18)

all_lines = [h, first_line, second_line, third_line]

def init():
    return all_lines

def animate(i):
    # get the latest data
    filtered_data = data[abs(data.epoch - all_epochs[i]) < 1]
    u_pred = filtered_data.u_pred.to_numpy()
    U_pred = griddata(X_star.T, u_pred.flatten(), (X, T), method='cubic')

    h.set_array(U_pred.T)
    first_line.set_data(x, U_pred[0, :])
    second_line.set_data(x, U_pred[128, :])
    third_line.set_data(x, U_pred[-10, :])
    ax_top.set_title('$u(t,x)$ - ' + str(int(all_epochs[i])) + ' epochs', fontsize=18) 
    all_lines = [h, first_line, second_line, third_line, ax_top]

anim = animation.FuncAnimation(fig,
                               animate,
                               frames=len(all_epochs),
                               interval=100,
                               blit=False) 

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
anim.save(sys.argv[1][:-3] + 'mp4', writer=writer)

# plt.savefig(filename, dpi=300)
# plt.close()
