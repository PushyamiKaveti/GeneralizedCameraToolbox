#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:59:38 2020

@author: Pushyami Kaveti
"""
import numpy as np
from helper_functions import helper_functions

class Camera:
    '''
    Class for a central projection camera model. 
    '''
    def __init__(self, focal=6*1e-3, resolution = (640,480), pp = (320,240), pix_siz = 5*1e-6, rot=np.eye(3), trans=np.zeros(3,1), noise=0):
        self.f=focal
        self.res = resolution
        self.cx = pp[0]
        self.cy = pp[1]
        self.pixel_size = pix_siz
        self.poseR = rot
        self.poset = trans
        self.R = rot.transpose()
        self.t = -self.R * trans
        self.K = np.array([[self.f/self.pixel_size, 0, self.cx] , [0 , self.f/self.pixel_size , self.cy ],[ 0 , 0 , 1]])
        self.P = helper_functions.compose_T(self.R, self.t) * self.K
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
        return np.linalg.inv(self.K) * p
    
    def project(self, P):
        '''
        Method to projectr 3D world points
        to image plane of the current camera
        Parameters
        ----------
        p : 3D world points 3 X N

        Returns
        -------
        the 2D image coordinates

        '''
        P = helper_functions.euclid_to_homo(P)
        p = self.T * P
        p = helper_functions.homo_to_euclid(p)
        p = p + np.eye(2)*np.random.randn(p.shape)
        return p
    
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
        self.poseR = self.poseR  * rot
        self.poset = -self.poseR * trans + self.poset
        self.R = self.poseR.transpose()
        self.t = -self.R * self.poset
        self.P = helper_functions.compose_T(self.R, self.t) * self.K