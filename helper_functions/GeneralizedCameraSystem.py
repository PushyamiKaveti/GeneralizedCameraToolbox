# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:10:08 2020

@author: Pushyami Kaveti
"""
import numpy as np
from helper_functions import helper_functions
from helper_functions import camera

class GeneralizedCamera:
    def __init__(self, num, cams, rot=np.eye(3), trans=np.zeros((3,1))):
        self.num_cams = num
        self.cams = cams
        self.poseR = rot
        self.poset = trans
        self.R = rot.transpose()
        self.t = -1*self.R @ trans

    def build_camera_system(self, yaml_file):
        pass
    
    def project(self, P):
        '''
         Method to projectr 3D world points
        to image plane of the current camera
        Parameters
        ----------
        p : 3D world points 3 X N 

        Returns
        -------
        p_all : 2D images coordinates of size num_cams X 2 X N 
        masks : a boolean array of dimension num_cams X 1 X N indicating valid projects
        '''
        #convert the 3D point from world into the generalized camera coordinate frame
        tmp =  helper_functions.compose_T(self.R, self.t.T)[:-1, :]
        P = helper_functions.euclid_to_homo(P)
        P_local = tmp @ P
        N = P.shape[1]
        masks = np.zeros((self.num_cams,1,N))
        p_all = np.zeros((self.num_cams,2,N))
        for i, c in enumerate(self.cams):
            p_all[i], masks[i] = c.project(P_local)
        return p_all, masks
    
    def get_plucker_coords(self,p):
        pass
    
    def plot_image(self, P, axs, *args, **kwargs):
        '''
        Method to project the 3D points into image and plot
        the image coordinates on a grid of dims (width, height)
        Parameters
        ----------
        P : 3XN 3D points with respect to world
        ax : axes object of the plot
        Returns
        -------
        p_all : 2D images coordinates of size num_cams X 2 X N 
        masks : a boolean array of dimension num_cams X 1 X N indicating valid projects

        '''
        tmp =  helper_functions.compose_T(self.R, self.t.T)[:-1, :]
        P = helper_functions.euclid_to_homo(P)
        P_local = tmp @ P
        N = P.shape[1]
        masks = np.zeros((self.num_cams,1,N))
        p_all = np.zeros((self.num_cams,2,N))
        for i, c in enumerate(self.cams):
            axs[0, i].set_title('Axis [0, '+str(i)+']')
            p_all[i], masks[i] = c.plot_image(P_local, axs[0,i],*args, **kwargs)
        return p_all, masks
    
    def plot_camera(self, ax,  *args, **kwargs):
        '''
        Method to plot the generalized camera array

        Parameters
        ----------
        ax : axis object which should have been initialized before this call
        Returns
        -------
        None.

        '''
        # get transforms of camera array
        T_array = np.zeros((0, 4,4))
        for c in self.cams:
            T_cam = helper_functions.compose_T(c.poseR, c.poset.T)
            T_array = np.append(T_array,T_cam[None], axis=0)
        
        # Plot camera array at pos 1
        wTc = helper_functions.compose_T(self.poseR, self.poset.T)
        helper_functions.plot_cam_array(ax, T_array, wTc,  *args, **kwargs)

    def transform(self, rot, trans):
        '''
        Method to transform the current Generliazed camera
        to a new pose
        
        Parameters
        ----------
        rot : Rotation matrix 3x3
        trans : translation vbector 3 x 1

        Returns
        -------
        None.

        '''
        self.poset = self.poseR @ trans + self.poset
        self.poseR = self.poseR  @ rot
        self.R = self.poseR.transpose()
        self.t = -1 * self.R @ self.poset
        
        