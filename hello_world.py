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
helper_functions.set_axes_equal(ax1)

T1 = np.eye(4)
helper_functions.plot_camera(ax1, T1, f=1.0)

T2 = np.array([[  0.7071068,  0.0000000, -0.7071068, 5.0],
               [  0.5000000,  0.7071068,  0.5000000, 0.1],
               [  0.5000000, -0.7071068,  0.5000000, 0.0],
               [  0.0      ,  0.0      ,  0.0      , 1.0]])

helper_functions.plot_camera(ax1, T2, f=1.0)

plt.pause(.5)


