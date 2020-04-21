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
    