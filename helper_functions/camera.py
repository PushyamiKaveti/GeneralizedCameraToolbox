#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:59:38 2020

@author: Pushyami Kaveti
"""
import numpy as np
from helper_functions import helper_functions
from helper_functions import plucker_functions

class Camera:
    '''
    Class for a central projection camera model. 
    '''
    def __init__(self, focal=6*1e-3, resolution = (640,480), pp = (320,240), pix_siz = (5*1e-6,5*1e-6) , rot=np.eye(3), trans=np.zeros((3,1)), noise=0.01):
        self.f=focal
        self.res = np.array([resolution[0], resolution[1]])
        self.cx = pp[0]
        self.cy = pp[1]
        self.pixel_size = np.array([pix_siz[0], pix_siz[1]])
        self.poseR = rot
        self.poset = trans
        self.R = rot.transpose()
        self.t = -1*self.R @ trans
        self.K = np.array([[self.f/self.pixel_size[0], 0, self.cx], 
                           [0 , self.f/self.pixel_size[1] , self.cy],
                           [ 0 , 0 , 1]])
        tmp =  helper_functions.compose_T(self.R, self.t.transpose())[:-1, :]
        self.P = self.K @ tmp
        self.noise = noise
    
    def get_normalized_coords(self, p):
        '''
        Method to convert the image 
        coordinates to normalized coordinates 
        in camera coordinate system.
        Parameters
        ----------
        p : image coordinates (2 x 1 or 2 x N)

        Returns
        -------
        

        '''
        p = helper_functions.euclid_to_homo(p)
        #return np.linalg.inv(self.K) @ p
        p = p.astype('float32')
        p[0, :] = (p[0, :] - self.K[0,2]) / self.K[0,0]
        p[1, :] = (p[1, :] - self.K[1,2]) / self.K[1,1]
        return p
    
    def get_fov(self):
        '''
        Returns
        -------
        the horizontal and vertical field of view in radians

        '''
        return 2 * np.arctan(self.res * self.pixel_size /(2* self.f))

    def project(self, P):
        '''
        Method to projectr 3D world points
        to image plane of the current camera
        Parameters
        ----------
        p : 3D world points 3 X N

        Returns
        -------
        p : 2D images coordinates fo size 2XN 
        mask : a boolean array of dimension 1 X N indicating valid projects

        '''
        P = helper_functions.euclid_to_homo(P)
        p = self.P @ P
        p = helper_functions.homo_to_euclid(p)
        p = p + self.noise * np.eye(2) @ np.random.randn(p.shape[0], p.shape[1])
        tmp=self.noise * np.eye(2) @ np.random.randn(p.shape[0], p.shape[1])
        #check bounds of the points and make then -1 if they are out of bounds
        mask = ((0 <= p[0]) & (p[0] < self.res[0])) & ((0 <= p[1]) & (p[1] < self.res[1]))
        return p , mask
    
    def transform(self, rot, trans):
        '''
        Method to transform the current camera
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
        tmp = helper_functions.compose_T(self.R, self.t.T)[:-1, :]
        self.P = self.K @ tmp 

    def plot_image(self, P, ax, *args, **kwargs):
        '''
        Method to project the 3D points into image and plot
        the image coordinates on a grid of dims (width, height)
        Parameters
        ----------
        P : 3XN 3D points with respect to world
        ax : axes object of the plot
        Returns
        -------
        p : 2D images coordinates fo size 2XN 
        mask : a boolean array of dimension 1 X N indicating valid projects

        '''
        ax.set_xlim(0, self.res[0])
        ax.set_ylim(0, self.res[1])
        ax.grid(True)
        p, mask = self.project(P)
        pp = p[:,mask]
        ax.plot(pp[0, :], pp[1, :], *args, **kwargs)
        return p, mask
    
    def get_plucker_coords(self,p):
        '''
        Method to convert the 2D image coordinates into 6D plucker vectors
        in the camera coordinate frame. For a central camera model where the
        pinhole is located at the origin,moment is always zero
        Parameters
        ----------
        p : 2XN image coordinates
        Returns
        -------
        PL : 6 X N plucker coordinates of p each element of the form (q,m)
        '''
        pp = self.get_normalized_coords(p)
        m = np.zeros((3,p.shape[1]))
        pp = np.vstack((pp,m))
        return pp
        
        