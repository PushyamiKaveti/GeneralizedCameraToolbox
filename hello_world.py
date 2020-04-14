#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:48:33 2020

@author: vik748
"""

import numpy as np
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
np.set_printoptions(precision=8,suppress=True)
from helper_functions import helper_functions
#from transforms3d.euler import euler2mat, mat2euler
import argparse


fig1,ax1 = helper_functions.initialize_3d_plot(number=1, limits = np.array([[-10,10],[-10,10],[-10,10]]), view=[-30,-90])

wP1 = np.array([1,-1,7.5])


# Define camera array
T_cam1 = np.eye(4)
T_cam2 = np.eye(4)
T_cam2[0,-1] = 2.5

T_array = np.zeros((0, 4,4))
T_array = np.append(T_array,T_cam1[None], axis=0)
T_array = np.append(T_array,T_cam2[None], axis=0)

# Plot camera array at pos 1
wTa1 = np.eye(4)
helper_functions.plot_cam_array(ax1, T_array, wTa1, color = 'orange')

# Plot camera array at pos 2
wTa2 = np.array([[  0.7071068,  0.0000000, -0.7071068, 5.0],
                 [  0.5000000,  0.7071068,  0.5000000, 0.1],
                 [  0.5000000, -0.7071068,  0.5000000, 0.0],
                 [  0.0      ,  0.0      ,  0.0      , 1.0]])
helper_functions.plot_cam_array(ax1, T_array, wTa2, color = 'cyan')


# Draw infinite vectors through a point
X0 = np.array([0,0,0])
X0 = wTa2[:3,-1]
V1 = wP1
V2 = wP1 - wTa2[:3,-1]

ax1.plot(wP1[[0]],wP1[[1]],wP1[[2]],'*', markersize=6)
helper_functions.plot_3Dvector(ax1, V1, origin = np.array([0,0,0]))
helper_functions.plot_3Dvector(ax1, V2, origin = wTa2[:3,-1])

#plt.pause(.5)
plt.show()

