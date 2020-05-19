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
    def select_correspondences(self, plucker1, plucker2, masks_1,masks_2):
        num_cams = plucker1.shape[0]
        corrs1=np.zeros((6,0))
        corrs2=np.zeros((6,0))
       
        #for camera i
        for i in range(num_cams):
            # find the correspondences using mask with every other camera
            corrs_mask = masks_1[i] & masks_2
            
            # Make the correspondence mask with itself as false. Coz we
            #donot want to track the landmark in the same camera in next frame
            # the final corrs_mask contans True where camera i 
            # has a correspondence in camera k to e=a particular landmark j
            
            #rows are camera indices and columns are landmarks
            for ii in range(i+1):
                corrs_mask[ii,:, :] = False
            # for landmark j
            for j in range(masks_1.shape[2]):
                ch =[]
                # in all the rows of corrs_mask for landmark j find where it is True.
                # the indices of True gives us the cameras it has correspondences with
                tmp=np.argwhere(corrs_mask[:,0,j]==True)
                # If none of the cameras have correspondences move on to next landmark
                if len(tmp) == 0:
                    continue
                # if only one correspondence is found store it in ch list
                elif len(tmp) == 1:
                    ch = [tmp[0,0]]
                else:
                    # if more than one correspondences is found choose two out of them
                    ch = np.random.choice(tmp[:,0],2,replace=False)
                # for eachof the chosen correspondence camera index add the corresponding plucker vetors
                for c_ind in ch:
                    # plucker vector of camera i and landmark j
                    corrs1 = np.append(corrs1, plucker1[i, : , [j]].T, axis=1)
                    #plucker vector of camera c_ind and landmark j
                    corrs2 = np.append(corrs2, plucker2[c_ind, : , [j]].T, axis=1)
        return corrs1, corrs2
    
    def compute_essential_central(self, corrs1_img, corrs2_img, c):
        #corrs1  = c.get_normalized_coords(corrs1_img)
        #corrs2  = c.get_normalized_coords(corrs2_img)
        from helper_functions import helper_functions
        corrs1 = helper_functions.euclid_to_homo(corrs1_img)
        corrs2 = helper_functions.euclid_to_homo(corrs2_img)
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
    
    def solve_gec(self, corrs1, corrs2,corrs1_img,corrs2_img):
        '''
        
        Parameters
        ----------
        corrs1 : TYPE
            DESCRIPTION.
        corrs2 : TYPE
            DESCRIPTION.

        Returns
        -------
        R_final : TYPE
            DESCRIPTION.
        t_sol : TYPE
            DESCRIPTION.

        '''
        assert corrs1.shape == corrs2.shape, "the correspondences do not have same size"
        num_corrs = corrs1.shape[1]
        
        
        #Build the data matrix with Ae and Ar components
        Ae = np.zeros((num_corrs , 9))
        Ar = np.zeros((num_corrs , 9))
        
        for i in range(num_corrs):
            Ae[i, :] = np.kron(corrs2[0:3, i], corrs1[0:3,i])
            Ar[i, :] = np.kron(corrs2[3:6, i], corrs1[0:3,i]) + np.kron(corrs2[0:3, i], corrs1[3:6,i])
            
        #basic method
        Ar_pinv= np.linalg.pinv(Ar)
        # tmp = (AA+ - I)
        tmp = Ar @ Ar_pinv - np.eye(num_corrs)
        #B = (AA+ - I) Ae
        B = tmp @ Ae
        #print("rank of B :" + str(np.linalg.matrix_rank(B)))
        #perform SVD on the new data matrix
        U_g, D_g, Vh_g = np.linalg.svd(B)
        
        # get the vector corresponding to the smallest singulat vector in V
        # i.e the last row or vh coz vh = V.T
        V_s_g = Vh_g[8, :]
        E_hat_g = V_s_g.reshape((3,3))
        
        # Perform another svd on E_hat and force the diagonal singular values to be equal and 1
        U_hat_g, D_hat_g, V_hat_h_g = np.linalg.svd(E_hat_g)
        E_g = U_hat_g @ np.diag([1,1,0]) @ V_hat_h_g
        
        R1_g, R2_g, t_g = cv2.decomposeEssentialMat(E_g)
        #decompose the essential matrix to get R and t
        
        # from openGV 
        # solve the equations to get absolute t
        if np.linalg.det(R1_g) < 0:
            R1_g = -1 * R1_g
        if np.linalg.det(R2_g) < 0:
            R2_g = -1 * R2_g
        
        #openGV implementation for translation
        #Build the data matrix with Atrans1 and Atrans2 components
        # Atrans1 = np.zeros((num_corrs , 3))
        # Atrans2 = np.zeros((num_corrs , 3))
        
        # btrans1 = np.zeros((num_corrs , 1))
        # btrans2 = np.zeros((num_corrs , 1))
        
        # for jj in range(num_corrs):
        #     tmp = (corrs2[0:3,jj].T @ R1_g)
        #     Atrans1[jj, :] =  np.cross(tmp, corrs1[0:3, jj].T)
            
        #     tmp = (corrs2[0:3,jj].T @ R2_g)
        #     Atrans2[jj, :] =  np.cross(tmp, corrs1[0:3, jj].T)
            
        #     btrans1[jj] = -1 * np.dot(corrs1[0:3,jj] , R1_g.T @ corrs2[3:6, jj]) - np.dot(corrs1[3:6, jj], R1_g.T @ corrs2[0:3,jj])
        #     btrans2[jj] = -1 * np.dot(corrs1[0:3,jj] , R2_g.T @ corrs2[3:6, jj]) - np.dot(corrs1[3:6, jj], R2_g.T @ corrs2[0:3,jj])
        
        # #find pseudo iinverse of Atrans1 and Atrans2
        # Atrans1_pinv= np.linalg.pinv(Atrans1)
        # trans1 = Atrans1_pinv @ btrans1;
        # Atrans2_pinv= np.linalg.pinv(Atrans2)
        # trans2 = Atrans2_pinv @ btrans2;
        
        # residual1 = Atrans1 @ trans1 - btrans1
        # residual2 = Atrans2 @ trans2 - btrans2
        
        # if np.linalg.norm(residual1) < np.linalg.norm(residual2) :
        #     R_solution = R1_g.T
        #     t_solution = trans1
        # else:
        #     R_solution = R2_g.T
        #     t_solution = trans2
            
    ############################ OPEN GV IMPLEMENTATION DONE ############################
            
        b1 =  -1 * Ar @ R1_g.flatten().T
        tmp = R1_g @ corrs1[0:3,:]
        At1 = np.cross(tmp.T , corrs2[0:3,:].T)
        
        # # solve for translation At. t_final = b. dims : At = n X 3, t_final = 3 X 1 , b = n x 1
        fit1 = np.linalg.lstsq(At1, b1,rcond=None)
        t_sol1 = fit1[0]
        res1 = fit1[1]
        
        b2 =  -1 * Ar @ R2_g.flatten().T
        tmp=R2_g @ corrs1[0:3,:]
        At2 = np.cross(tmp.T , corrs2[0:3,:].T)
        
        # # solve for translation At. t_final = b. dims : At = n X 3, t_final = 3 X 1 , b = n x 1
        fit2 = np.linalg.lstsq(At2, b2, rcond=None)
        t_sol2 = fit2[0]
        res2 = fit2[1]
        
        if res1 < res2 :
            return R1_g , t_sol1
        else:
            return R2_g, t_sol2

        #ret_val, R_final, t_final, mask1 = cv2.recoverPose(E_g,corrs1_img.T,corrs2_img.T )

        # solve the equations to get absolute t
        #b =  -1 * Ar @ R_final.flatten().T
        #tmp=R_final @ corrs1[0:3,:]
        #At = np.cross(tmp.T , corrs2[0:3,:].T)
        
        # solve for translation At. t_final = b. dims : At = n X 3, t_final = 3 X 1 , b = n x 1
        #t_sol = np.linalg.lstsq(At, b)[0]
        #return R_final, t_sol
    
    def compute_essential_gencam_RH(self, plucker1, plucker2, masks_1,masks_2, genc):
        '''
        Method to solve the generalized epipolar constraint 
        and compute the relative translation and rotation based on 
        the algorithm defined by Richard AHrtley and huidong li

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
        corrs1, corrs2 = self.select_correspondences(plucker1, plucker2, masks_1,masks_2)
        corrs_mask=[]
        
        selected_cam =0
        for i in range(num_cams):
            corrs_mask = masks_1[i] & masks_2[i]
            no=len(corrs_mask[corrs_mask == True])
            if no > 10:
                selected_cam=i
                break
            
        corrs1_img = plucker1[selected_cam, 0:3 , corrs_mask[0,:]].T
        corrs2_img = plucker2[selected_cam, 0:3 , corrs_mask[0,:]].T
        corrs1_img = genc.cams[selected_cam].K @ corrs1_img
        corrs2_img = genc.cams[selected_cam].K @ corrs2_img
        corrs1_img = corrs1_img[0:2,:]
        corrs2_img = corrs2_img[0:2,:]
        
        R_final, t_sol = self.solve_gec(corrs1, corrs2, corrs1_img, corrs2_img)
        return R_final, t_sol
    
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

        corrs_mask=[]
        for i in range(num_cams):
            corrs_mask = masks_1[i] & masks_2[i]
            no=len(corrs_mask[corrs_mask == True])
            if no > 10:
                break
            
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
        
        # from openGV 
         # solve the equations to get absolute t
        # if np.linalg.det(R1_g) < 0:
        #     R1_g = -1 * R1_g
        # if np.linalg.det(R2_g) < 0:
        #     R2_g = -1 * R2_g
        # b1 =  -1 * Ar @ R1_g.flatten().T
        # tmp=R1_g @ corrs1[0:3,:]
        # At1 = np.cross(tmp.T , corrs2[0:3,:].T)
        
        # # solve for translation At. t_final = b. dims : At = n X 3, t_final = 3 X 1 , b = n x 1
        # t_sol1 = np.linalg.lstsq(At1, b1)[0]
        
        # b2 =  -1 * Ar @ R2_g.flatten().T
        # tmp=R2_g @ corrs1[0:3,:]
        # At2 = np.cross(tmp.T , corrs2[0:3,:].T)
        
        # # solve for translation At. t_final = b. dims : At = n X 3, t_final = 3 X 1 , b = n x 1
        # t_sol2 = np.linalg.lstsq(At2, b2)[0]
        
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
    