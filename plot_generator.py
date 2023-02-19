# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:26:38 2022

@author: CS
"""
from matplotlib import cm
import numpy as np
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
#import seaborn as sns
from matplotlib.colors import LightSource
from matplotlib import cbook
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
    

def show_heatmap(residue_hypotenus):
    plt.imshow(residue_hypotenus, cmap='hot', interpolation='nearest')
    plt.xlabel("Residues")
    plt.ylabel("Snapshots")
    plt.title('Hypotenus values')
    plt.colorbar()
    plt.show()

def snapshot_plots(se_residues):
    snap_errs=np.sqrt(np.mean(se_residues,axis=1))
    fig, ax = plt.subplots()
    #ax.plot(a, snap_errs, 'k--', label='All snaps')
    ax.plot(np.arange(0,11), snap_errs[0:11], 'r', label='Closed region')
    ax.plot(np.arange(11,25), snap_errs[11:25], 'g', label='Transition region')
    ax.plot(np.arange(25,100), snap_errs[25:100], 'b', label='Open region')
    
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame()
    plt.xlabel('Snapshots in test data ')
    plt.ylabel('RMSE')
    ax.grid()
    plt.show()
    
def residue_plots(se_residues):
    closed_rmse=np.sqrt(np.mean(se_residues[0:11],axis=0))
    transition_rmse=np.sqrt(np.mean(se_residues[11:25],axis=0))
    open_rmse=np.sqrt(np.mean(se_residues[25:100],axis=0))
    
    fig, ax = plt.subplots()
    #ax.plot(np.arange(0,437), all_residues, 'k:', label='All snapshots')
    ax.plot(np.arange(0,437), closed_rmse, 'r', label='Closed region', linewidth=0.8)
    ax.plot(np.arange(0,437), transition_rmse, 'g', label='Transition region',linewidth=0.8)
    ax.plot(np.arange(0,437), open_rmse, 'b', label='Open region',linewidth=0.8)
    
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')
    legend.get_frame()
    plt.xlabel('Residues ')
    plt.ylabel('RMSE')
    ax.grid()
    plt.show()

def LL_scatter(LL_pred,i,j,k):
    plt.style.use('_mpl-gallery')
    
    # Make data
    np.random.seed(19680801)

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    xs = LL_pred[0:251,i]
    ys = LL_pred[0:251,j]
    zs = LL_pred[0:251,k]
    ax.scatter(xs, ys, zs, s=4,c='r')
    
    xt = LL_pred[251:443,i]
    yt = LL_pred[251:443,j]
    zt = LL_pred[251:443,k]
    ax.scatter(xt, yt, zt, s=4,c='b')

    #ax.set(xticklabels=[],
    #   yticklabels=[],
    #   zticklabels=[])
    ax.set_xlabel('LL node'+str(i),fontsize='x-small')
    ax.set_ylabel('LL node'+str(j),fontsize='x-small')
    ax.set_zlabel('LL node'+str(k),fontsize='x-small')

    plt.show()

def LL_3dplot_train(LL_pred,i,j,k):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = LL_pred[0:251,i]
    ys = LL_pred[0:251,j]
    zs = LL_pred[0:251,k]
    
    xt = LL_pred[251:443,i]
    yt = LL_pred[251:443,j]
    zt = LL_pred[251:443,k]
    
    ax.scatter(xs, ys, zs, c='r', s=8,marker='o') #closed
    ax.scatter(xt, yt, zt, c='b',s=8, marker='^') #open
    
    ax.set_xlabel('LL node'+str(i),fontsize='x-small')
    ax.set_ylabel('LL node'+str(j),fontsize='x-small')
    ax.set_zlabel('LL node'+str(k),fontsize='x-small')
    plt.show()
    
def LL_3dplot_test(LL_pred,i,j,k):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = LL_pred[0:251,i]
    ys = LL_pred[0:251,j]
    zs = LL_pred[0:251,k]
    
    
    xt = LL_pred[251:443,i]
    yt = LL_pred[251:443,j]
    zt = LL_pred[251:443,k]
    
    xp = LL_pred[0:100,i]
    yp = LL_pred[0:100,j]
    zp = LL_pred[0:100,k]
    
    ax.scatter(xs, ys, zs, c='r', s=8,marker='o') #closed
    ax.scatter(xt, yt, zt, c='b',s=8, marker='^') #open
    ax.scatter(xp, yp, zp, c='y',s=8, marker='*') #open

    
    ax.set_xlabel('LL node'+str(i),fontsize='x-small')
    ax.set_ylabel('LL node'+str(j),fontsize='x-small')
    ax.set_zlabel('LL node'+str(k),fontsize='x-small')
    plt.show()
    
def plot_3d(rmse_residues):
   
    # defining axes
    def RMSE(snap, res):
        return(rmse_residues[snap][res])
    #RMSE = rmse_residues
    snap = np.arange(0,100)
    residue = np.arange(0,437)
    residue, snap = np.meshgrid(residue, snap)

    region = np.s_[0:100, 0:437]
    residue, snap, rmse_residues = residue[region], snap[region], rmse_residues[region]
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(rmse_residues, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(residue, snap, rmse_residues, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)
    
    plt.show()

# Save array to txt
# np.savetxt('./textfiles/closed_rmse.txt',)