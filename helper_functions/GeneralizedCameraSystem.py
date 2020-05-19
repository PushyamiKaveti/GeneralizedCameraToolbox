# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:10:08 2020

@author: Pushyami Kaveti
"""
import numpy as np
from helper_functions import helper_functions
from helper_functions import camera
import yaml

class GeneralizedCamera:
    def __init__(self, num=0, cams=[], rot=np.eye(3), trans=np.zeros((3,1))):
        self.num_cams = num
        self.cams = cams
        self.poseR = rot
        self.poset = trans
        self.R = rot.transpose()
        self.t = -1*self.R @ trans

    def build_camera_system(self, yaml_file, rot=np.eye(3), trans=np.zeros((3,1))):
        '''
        Method to read the kalibr yaml file and create 
        generalized camera object corresponding to that.
        This is for real camera not simulation

        Parameters
        ----------
        yaml_file : TYPE
            DESCRIPTION.
        rot : TYPE, optional
            DESCRIPTION. The default is np.eye(3).
        trans : TYPE, optional
            DESCRIPTION. The default is np.zeros((3,1)).

        Returns
        -------
        None.

        '''
        self.poseR = rot
        self.poset = trans
        self.R = rot.transpose()
        self.t = -1*self.R @ trans
        with open(yaml_file) as file:
            root = yaml.load(file)
            i = 0
            self.cams=[]
            for item, info in root.items():
                print("-------------------"+item+"----------------------")
                #extract distortion coefficients . this is for radtan model
                dist_c = info['distortion_coeffs']
                dist_c.append(0)
                
                #extract K matrix intrsinics
                intr = info['intrinsics']
                K = np.float64([[ intr[0] , 0, intr[2] ],
                              [ 0 , intr[1], intr[3] ],
                              [ 0 , 0 , 1 ]])
                #get the resolution
                res = info['resolution']
                
                if 'T_cn_cnm1' in info:
                    T = np.array(info['T_cn_cnm1'])
                    # get the R and t for each camera w.r.t the first one
                    R = T[0:3, 0:3] @ self.cams[i-1].R
                    t = T[0:3, 0:3] @ self.cams[i-1].t + T[0:3, 3:4]
                    
                    tmp = helper_functions.compose_T(R, t.T)
                    TT = helper_functions.T_inv(tmp)
                    c = camera.Camera(resolution = res, rot= TT[0:3, 0:3], trans=TT[0:3, 3:4])
                    c.K = K
                    c.cx = K[0, 2]
                    c.cy = K[1, 2]
                    c.dist_coeffs = np.float64(dist_c)
                    self.cams.append(c)
                                    
                else:
                    R = np.eye(3)
                    t = np.zeros((3,1))
                    c = camera.Camera(resolution = res, rot= R, trans=t)
                    c.K = K
                    c.cx = K[0, 2]
                    c.cy = K[1, 2]
                    c.dist_coeffs = np.float64(dist_c)
                    self.cams.append(c)
                i = i + 1
            self.num_cams = i
    
    def project(self, P):
        '''
         Method to project 3D world points
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
        masks = np.zeros((self.num_cams,1,N), dtype=bool)
        p_all = np.zeros((self.num_cams,2,N))
        for i, c in enumerate(self.cams):
            p_all[i], masks[i] = c.project(P_local)
        return p_all, masks
    
    def get_plucker_coords(self,p, normalize = True):
        '''
        Method to compute the plucker vectprs for the image points p
        in the generalized camera reference frame
        Parameters
        ----------
        p : num_cams X 2 X N image coordinates
        masks : num_cams X 1 X N boolean array representing valid image coordinates
        Returns
        -------
        PL_all : num_cams X 6 X N plucker coordinates of p in the 
        generalized camera reference frame

        '''
        N = p.shape[2]
        PL_all = np.zeros((self.num_cams,6,N))
        for i, c in enumerate(self.cams):
            #plucker vectors in camera ref frame
            pl_tmp = c.get_plucker_coords(p[i],normalize= normalize)
            #plucker vectors in generalized camera ref frame
            GE_tmp = np.zeros((6,6))
            GE_tmp[0:3, 0:3] = c.poseR
            GE_tmp[3:6, 3:6] = c.poseR
            GE_tmp[3:6, 0:3] = helper_functions.skew(c.poset[:,0]) @ c.poseR
            PL_all[i] = GE_tmp @ pl_tmp
        return PL_all
            
    
    def get_plucker_coords_cam(self, p,c, normalize=True):
        '''
        Method to compute the plucker vectors for the image points p
        in the generalized camera reference frame for a particular camera
        Parameters
        ----------
        p : 2 X N image coordinates
        c : index of the camera
        Returns
        -------
        PL : 6 X N plucker coordinates of p in the 
        generalized camera reference frame

        '''
        N = p.shape[1]
        PL = np.zeros((6,N))
        cam = self.cams[c]
        #plucker vectors in camera ref frame
        pl_tmp = cam.get_plucker_coords(p,normalize= normalize)
        #plucker vectors in generalized camera ref frame
        GE_tmp = np.zeros((6,6))
        GE_tmp[0:3, 0:3] = cam.poseR
        GE_tmp[3:6, 3:6] = cam.poseR
        GE_tmp[3:6, 0:3] = helper_functions.skew(cam.poset[:,0]) @ cam.poseR
        PL= GE_tmp @ pl_tmp
        return PL

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
        masks = np.zeros((self.num_cams,1,N), dtype=bool)
        p_all = np.zeros((self.num_cams,2,N))
        for i, c in enumerate(self.cams):
            axs[0, i].set_title('Axis [0, '+str(i)+']')
            p_all[i], masks[i] = c.plot_image(P_local, axs[0,i],*args, **kwargs)
        return p_all, masks
    
    def plot_camera(self, ax,  scale=1.0, cam_ind = -1, *args, **kwargs):
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
        cams_to_plot = self.cams
        if cam_ind != -1 and cam_ind < self.num_cams :
            cams_to_plot = [self.cams[cam_ind]]
            
        for c in cams_to_plot:
            T_cam = helper_functions.compose_T(c.poseR, c.poset.T)
            T_array = np.append(T_array,T_cam[None], axis=0)
        
        # Plot camera array at pos 1
        wTc = helper_functions.compose_T(self.poseR, self.poset.T)
        helper_functions.plot_cam_array(ax, T_array, wTc,  scale=scale, *args, **kwargs)

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
        
        