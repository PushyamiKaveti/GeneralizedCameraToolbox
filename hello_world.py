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
from helper_functions import plucker_functions as pf

def single_cam_demo():
    fig1,ax1 = helper_functions.initialize_3d_plot(number=1, limits = np.array([[-10,10],[-10,10],[-10,10]]), view=[-30,-90])

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
    wTa2 = np.array([[  0.86602, 0, -0.5, 4.0],
                     [ 0,  1,  0, 0],
                     [ 0.5, 0,0.86602, 0.0],
                     [  0.0 ,  0.0,  0.0 , 1.0]])
    
    #wTa2 = np.array([[  1, 0, 0, -2.0],
    #                 [  0, 1, 0, 0.0],
    #                 [  0, 0, 1, 0.0],
    #                 [  0, 0, 0, 1.0]])
    helper_functions.plot_cam_array(ax1, T_array, wTa2, color = 'cyan')
    
    
    ######################################################################
    # PLOT CAMERA and TRANSFORMATION #
    #####################################################################
    pp = helper_functions.generate_3D_points(200, x_c=0,y_c=0,z_c=10,rad=2)
    
    rot = np.array([[ 0.86602, 0, -0.5 ],
                     [  0,  1,  0],
                     [  0.5, 0,0.86602]])
    #rot = np.eye(3)
    t = np.array([[4.0],[0],[0]])
    
    c = camera.Camera()
    print(helper_functions.compose_T(c.poseR, c.poset.T))
    fig2, ax2 = helper_functions.initialize_2d_plot(number=2)
    
    points1, mask1 = c.plot_image(pp, ax2, 'bo', markersize=2)
    
    c.transform(rot, t)
    print(helper_functions.compose_T(c.poseR, c.poset.T))
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
#gec_solver
# Initialize all the plots 
fig3,ax3 = helper_functions.initialize_3d_plot(number=3, limits = np.array([[-10,10],[-10,10],[-10,10]]), view=[-30,-90])
fig4, ax4 = helper_functions.initialize_2d_plot_multi(number=4,num_rows=1, num_cols=5)

# create the 3D points and plot them
pp = helper_functions.generate_3D_points(20, x_c=2,y_c=0,z_c=20,rad=5)
helper_functions.plot_3d_points(ax3, pp.T, None, 'bo', markersize=2)

# define a two camera array
# define cameras within the generalized camera frame
c1 = camera.Camera() #identity transform
rot2 = np.eye(3)
t2 = np.array([[2.5],[0],[0]])
c2 = camera.Camera(rot=rot2, trans=t2) # rotation of c2 wrt to the generalized camera frame for now it is c1
rot3 = np.eye(3)
t3 = np.array([[1.0],[2.0],[0]])
c3 = camera.Camera(rot=rot3, trans=t3)
rot4 = np.eye(3)
t4 = np.array([[4.5],[-1.0],[0]])
c4 = camera.Camera(rot=rot4, trans=t4)
rot5 = np.eye(3)
t5 = np.array([[7],[-2],[0]])
c5 = camera.Camera(rot=rot5, trans=t5)

genC = genCam.GeneralizedCamera(5, [c1,c2, c3, c4, c5])
genC.plot_camera(ax3, color="orange")

# project the 3D points into position one get the image points
points_gen1, masks_1 = genC.plot_image(pp, ax4, 'bo', markersize=2)

# Get the plucker vectors 
plucker1 = genC.get_plucker_coords(points_gen1)


ext_R = np.array([[ 0.86602, 0, -0.5 ],
                 [  0,  1,  0],
                 [  0.5, 0,0.86602]])
ext_t = np.array([[4.0],[0],[0]])

c11 = camera.Camera() #identity transform
rot2 = np.eye(3)
t2 = np.array([[2.5],[0],[0]])
c22 = camera.Camera(rot=rot2, trans=t2) # rotation of c2 wrt to the generalized camera frame for now it is c1
rot3 = np.eye(3)
t3 = np.array([[1.0],[2.0],[0]])
c33 = camera.Camera(rot=rot3, trans=t3)
rot4 = np.eye(3)
t4 = np.array([[4.5],[-1.0],[0]])
c44 = camera.Camera(rot=rot4, trans=t4)
rot5 = np.eye(3)
t5 = np.array([[7],[-2],[0]])
c55 = camera.Camera(rot=rot5, trans=t5)
genC2 = genCam.GeneralizedCamera(5, [c11,c22, c33, c44, c55])

#genC2.plot_camera(ax3, color="orange")

genC2.transform(ext_R, ext_t)
genC2.plot_camera(ax3, color="blue")

points_gen2, masks_2 = genC2.plot_image(pp, ax4, 'ro', markersize=2)
# Get the plucker vectors 
plucker2 = genC2.get_plucker_coords(points_gen2)



gec_solver = gec.GECSolver()
R_final, t_final = gec_solver.compute_essential_gencam_RH(plucker1, plucker2, masks_1, masks_2, genC)
TT =helper_functions.compose_T(R_final, t_final[np.newaxis,:])
TT = helper_functions.T_inv(TT)

print(TT)
genC.transform(TT[0:3, 0:3], TT[0:3, 3:4])
genC.plot_camera(ax3, color="magenta")

###################### Get the 3D points ###########################

# get the correspondening plucker vectors 
num_cams = points_gen1.shape[0]
corrs1=np.zeros((6,0))
corrs2=np.zeros((6,0))

for i in range(num_cams):
    corrs_mask = masks_1[i] & masks_2[i]
    corrs1 = np.append(corrs1, plucker1[i, : , corrs_mask[0,:]].T, axis=1)
    corrs2 = np.append(corrs2, plucker2[i, : , corrs_mask[0,:]].T, axis=1)
# transform the corrs2 into the first camera reference frame using the calculated pose
#plucker vectors in generalized camera ref frame
GE_tmp = np.zeros((6,6))
GE_tmp[0:3, 0:3] = TT[0:3,0:3]
GE_tmp[3:6, 3:6] = TT[0:3, 0:3]
GE_tmp[3:6, 0:3] = helper_functions.skew(TT[0:3 , -1]) @ TT[0:3, 0:3]
corrs2_trans = GE_tmp @ corrs2
 
# find the point of intersection
pluckerobj = pf.Plucker(1,2,typ="nothg")
recons_pts = np.zeros((3,0))
for i in range(corrs1.shape[1]):
    pl1 = pf.Plucker( corrs1[0:3, i],corrs1[3:6, i], typ="dir-moment")
    pl2 = pf.Plucker( corrs2_trans[0:3, i],corrs2_trans[3:6, i], typ="dir-moment")
    recons_pts = np.append(recons_pts , pluckerobj.point_of_intersection(pl1, pl2)[:, np.newaxis],  axis=1)
    #print(pt)
helper_functions.plot_3d_points(ax3, recons_pts.T, None, 'ro', markersize=5)
# plot the 3D points

plt.show()


