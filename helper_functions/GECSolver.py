# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:50:10 2020

@author: Pushyami Kaveti
"""

import numpy as np
import cv2

class GECSolver:
    def __init__(self):
        #initialize R and t and ny other params
        print("initiated GEC Solver")
    
    def create_mat_A(self):
        #take the correspondences and create the
        #data matrix to form a linear equation system
        pass
    
    def solve(self, corrs1, corrs2):
        #form the matrix form using the correspondences
        #perform normal SVD and get the E and R from it
        pass
    
    def compute_essential_central(self, corrs1_img, corrs2_img, c):
        corrs1  = c.get_normalized_coords(corrs1_img)
        corrs2  = c.get_normalized_coords(corrs2_img)
        assert corrs1.shape == corrs2.shape, "the correspondences do not have same size"
        num_corrs = corrs1.shape[1]
        A = np.zeros((num_corrs , 9))
        for i in range(num_corrs):
             A[i, :] = np.kron(corrs2[:, i], corrs1[:,i])
        #perform SVD on the data matrix
        U, D, Vh = np.linalg.svd(A)
        # get the vector corresponding to the smallest singulat vector in V
        # i.e the last row or vh coz vh = V.T
        V_s = Vh[8, :]
        E_hat = V_s.reshape((3,3))
        #print("E_hat : \n")
        #print(E_hat)
        # Perform another svd on E_hat and force the diagonal singular values to be equal and 1
        U_hat, D_hat, V_hat_h = np.linalg.svd(E_hat)
        E = U_hat @ np.diag([1,1,0]) @ V_hat_h
        
        #decompose the essential matrix to get R and t
        ret_val, R, t, mask1 = cv2.recoverPose(E,corrs1_img.T,corrs2_img.T )
        return E,R,t
    
    def compute_essential_gencam(self, plucker1, plucker2, masks_1,masks_2, genc):
        '''
        Method to solve the generalized epipolar constraint 
        and compute the relative translation and rotation

        Parameters
        ----------
        plucker1 : 
        plucker2 : 
        masks_1 : 
        masks_2 :
        genc :

        Returns
        -------
        None.

        '''
        num_cams = plucker1.shape[0]
        corrs1=np.zeros((6,0))
        corrs2=np.zeros((6,0))
        for i in range(num_cams):
            corrs_mask = masks_1[i] & masks_2[i]
            corrs1 = np.append(corrs1, plucker1[i, : , corrs_mask[0,:]].T, axis=1)
            corrs2 = np.append(corrs2, plucker2[i, : , corrs_mask[0,:]].T, axis=1)
            
        corrs_mask = masks_1[0] & masks_2[0]
        corrs1_img = plucker1[0, 0:3 , corrs_mask[0,:]].T
        corrs2_img = plucker2[0, 0:3 , corrs_mask[0,:]].T
        corrs1_img = genc.cams[0].K @ corrs1_img
        corrs2_img = genc.cams[0].K @ corrs2_img
        corrs1_img = corrs1_img[0:2,:]
        corrs2_img = corrs2_img[0:2,:]
        
        assert corrs1.shape == corrs2.shape, "the correspondences do not have same size"
        num_corrs = corrs1.shape[1]
        #Build the data matrix with Ae and Ar components
        Ae = np.zeros((num_corrs , 9))
        Ar = np.zeros((num_corrs , 9))
        
        for i in range(num_corrs):
            Ae[i, :] = np.kron(corrs2[0:3, i], corrs1[0:3,i])
            Ar[i, :] = np.kron(corrs2[3:6, i], corrs1[0:3,i]) + np.kron(corrs2[0:3, i], corrs1[3:6,i])
            
        #basic method
        #Horizontally stack Ae and Ar to form num_corrs X 18  matrix
        A_gen = np.hstack((Ae,Ar))
        #perform SVD on the data matrix
        U_g, D_g, Vh_g = np.linalg.svd(A_gen)
        # get the vector corresponding to the smallest singulat vector in V
        # i.e the last row or vh coz vh = V.T
        V_s_g = Vh_g[17, :]
        E_hat_g = V_s_g[0:9].reshape((3,3))
        R_hat_g = V_s_g[9:].reshape((3,3))
        #print("E_hat : \n")
        #print(E_hat)
        # Perform another svd on E_hat and force the diagonal singular values to be equal and 1
        U_hat_g, D_hat_g, V_hat_h_g = np.linalg.svd(E_hat_g)
        E_g = U_hat_g @ np.diag([1,1,0]) @ V_hat_h_g
        
        R1_g, R2_g, t_g = cv2.decomposeEssentialMat(E_g)
        #decompose the essential matrix to get R and t

        ret_val, R_final, t_final, mask1 = cv2.recoverPose(E_g,corrs1_img.T,corrs2_img.T )

        # solve the equations to get absolute t
        b =  -1 * Ar @ R_final.flatten().T
        tmp=R_final @ corrs1[0:3,:]
        At = np.cross(tmp.T , corrs2[0:3,:].T)
        
        # solve for translation At. t_final = b. dims : At = n X 3, t_final = 3 X 1 , b = n x 1
        t_sol = np.linalg.lstsq(At, b)[0]
        return R_final, t_sol
    
    def get_four_solutions(self, E):
        # get all four solutions
        U, D, Vt = np.linalg.svd(E)
        Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        #E1 = t1R1
        t1 = U @ Z @ U.T
        R1 = U @ W @ Vt
        
        #E2 = t2 R1
        t2 = U @ Z.T @ U.T
        #E3 = t1 R2
        R2 = U @ W.T @ Vt
        #E3 = t2 R2
        print("R1 :\n" )
        print(R1)
        print("R2 :\n")
        print(R2)
        print("t1:\n")
        print(t1)
        print("t2\n")
        print(t2)
        return R1, R2, t1, t2
    