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
from helper_functions import camera
import cv2
from helper_functions import GECSolver as gec
from helper_functions import GeneralizedCameraSystem as genCam

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
wTa2 = np.array([[  0.86602, 0, 0.5, -5.0],
                 [ 0,  1,  0, 0],
                 [ -0.5, 0,0.86602, 0.0],
                 [  0.0 ,  0.0,  0.0 , 1.0]])
helper_functions.plot_cam_array(ax1, T_array, wTa2, color = 'cyan')


# Draw infinite vectors through a point
#X0 = np.array([0,0,0])
#X0 = wTa2[:3,-1]
#V1 = wP1
#V2 = wP1 - wTa2[:3,-1]

#ax1.plot(wP1[[0]],wP1[[1]],wP1[[2]],'*', markersize=6)
#helper_functions.plot_3Dvector(ax1, V1, origin = np.array([0,0,0]))
#helper_functions.plot_3Dvector(ax1, V2, origin = wTa2[:3,-1])



######################################################################
# PLOT CAMERA and TRANSFORMATION #
#####################################################################
pp = helper_functions.generate_3D_points(50, x_c=0,y_c=0,z_c=10,rad=2)

rot = np.array([[ 0.86602, 0, 0.5 ],
                 [  0,  1,  0],
                 [  -0.5, 0,0.86602]])
t = np.array([[-5.0],[0],[0]])

c = camera.Camera()

fig2, ax2 = helper_functions.initialize_2d_plot(number=2)

points1, mask1 = c.plot_image(pp, ax2, 'bo', markersize=2)

c.transform(rot, t)

points2, mask2 = c.plot_image(pp, ax2,'ro', markersize=2)

helper_functions.plot_3d_points(ax1, pp.T, None, 'bo', markersize=2)

# find the point correspondences 
corrs_mask = mask1 & mask2
corrs1 = points1[: , corrs_mask]
corrs2 = points2[: , corrs_mask]

#perform a normal essential matrix calculation
gec_solver = gec.GECSolver()
E ,R, t = gec_solver.compute_essential_central(corrs1, corrs2, c)


# #decompose the essential matrix to get R and t
corrs_img1 = points1[: , corrs_mask] 
corrs_img2 = points2[: , corrs_mask] 
E_cv, mask = cv2.findEssentialMat(corrs_img1.T,corrs_img2.T,c.K)
ret_val, R_cv, t_cv, mask1 = cv2.recoverPose(E_cv,corrs_img1.T,corrs_img2.T )
print("essential cv                              Rotation CV                              translation CV   \n")
print(np.array2string(E_cv[0,:]) +"        "+np.array2string(R_cv[0,:]) + "       " + np.array2string(t_cv[0,:]))
print(np.array2string(E_cv[1,:]) +"        "+np.array2string(R_cv[1,:]) + "       " + np.array2string(t_cv[1,:]))
print(np.array2string(E_cv[2,:]) +"        "+np.array2string(R_cv[2,:]) + "       " + np.array2string(t_cv[2,:]))


print("\n\n\n")

# pass the correspondences into GEC solver and estimate R and T using the essential matrix algo
# plot the estimated R and T
T_cam_est = helper_functions.compose_T(R,t.T)
T_cam_est = helper_functions.T_inv(T_cam_est)
print("essential                                 Rotation                                 translation      \n")
print(np.array2string(E[0,:]) +"        "+np.array2string(R[0,:]) + "       " + np.array2string(t[0,:]))
print(np.array2string(E[1,:]) +"        "+np.array2string(R[1,:]) + "       " + np.array2string(t[1,:]))
print(np.array2string(E[2,:]) +"        "+np.array2string(R[2,:]) + "       " + np.array2string(t[2,:]))

helper_functions.plot_cam_array(ax1, T_array, T_cam_est, color = 'red')



###################################################################
#     PLOT THE CAMERA ROTATION AND TRANSLATION in a camera array  #
###################################################################
fig3,ax3 = helper_functions.initialize_3d_plot(number=3, limits = np.array([[-10,10],[-10,10],[-10,10]]), view=[-30,-90])

# define a two camera array
#rot = np.array([[ 0.86602, 0, 0.5 ],
#                 [  0,  1,  0],
#                 [  -0.5, 0,0.86602]])


# define cameras within the generalized camera frame
c1 = camera.Camera() #identity transform
rot1 = np.eye(3)
t1 = np.array([[2.5],[0],[0]])
c2 = camera.Camera(rot=rot1, trans=t1) # rotation of c2 wrt to the generalized camera frame for now it is c1

genC = genCam.GeneralizedCamera(2, [c1,c2])
#genC.plot_camera(ax3, color="orange")
fig4, ax4 = helper_functions.initialize_2d_plot_multi(number=4)

points1, mask1 = genC.plot_image(pp, ax4, 'bo', markersize=2)



ext_R = np.array([[ 0.86602, 0, 0.5 ],
                 [  0,  1,  0],
                 [  -0.5, 0,0.86602]])
ext_t = np.array([[-5.0],[0],[0]])

c11 = camera.Camera() #identity transform
rot11 = np.eye(3)
t11 = np.array([[2.5],[0],[0]])
c22 = camera.Camera(rot=rot11, trans=t11) # rotation of c2 wrt to the generalized camera frame for now it is c1
genC2 = genCam.GeneralizedCamera(2, [c11,c22])
genC2.plot_camera(ax3, color="orange")
genC2.transform(ext_R, ext_t)
genC2.plot_camera(ax3, color="blue")
points1, mask1 = genC2.plot_image(pp, ax4, 'ro', markersize=2)

2

plt.show()

