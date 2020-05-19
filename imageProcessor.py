# -*- coding: utf-8 -*-

"""
Created on Mon Apr 29 15:50:30 2020

@author: Pushyami Kaveti
"""

import cv2
import numpy as np
import yaml
from helper_functions import GECSolver as gec
from helper_functions import GeneralizedCameraSystem as genCam
from helper_functions import plucker_functions as pf
from helper_functions import helper_functions as helper

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
np.set_printoptions(precision=8,suppress=True)
import os
from helper_functions import zernike 


#############################################################################################
#                                   VISUALIZING FUNCTIONS                                   #
#############################################################################################
def displayMatches(img_left,kp1,img_right,kp2, matches, mask):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    '''
    #bool_mask = mask.astype(bool)
    img_valid = cv2.drawMatches(img_left,kp1,img_right,kp2,matches, None, 
                                matchColor=(0, 255, 0), 
                                matchesMask=None, flags=2)
    img_all = cv2.drawMatches(img_left,kp1,img_right,kp2,matches, img_valid, 
                              matchColor=(255, 0, 0), 
                              matchesMask=None, 
                              flags=1)
    return img_all

def displayMatches_manual(img_left,kp1,img_right,kp2):
    assert kp1.shape == kp2.shape, "wrong key point size"
    kp1 = kp1.astype(int)
    kp2 = kp2.astype(int)
    image = np.concatenate((img_left, img_right), axis=1)
    for i in range(kp1.shape[0]):
        cv2.line(image, (kp1[i][0], kp1[i][1]), (kp2[i][0], kp2[i][1]), (255, 255, 255), 2)
    return image
 
# First match 2 against 1
    matches_knn = matcher.knnMatch(des2,des1, k=2)
    
    matches = []
    # Run lowes filter and filter with difference higher than threshold this might
    # still leave multiple matches into 1 (train descriptors)
    # Create mask of size des1 x des2 for permissible matches
    mask = np.zeros((des1.shape[0],des2.shape[0]),dtype='uint8')
    for match in matches_knn:
        if len(match)==1 or (len(match)>1 and match[0].distance < threshold*match[1].distance):
                matches.append(match[0])
                mask[match[0].trainIdx,match[0].queryIdx] = 1
    
    # run matches again using mask but from 1 to 2 which should remove duplicates            
    # This is basically same as running cross match after lowe ratio test
    matches_cross = matcher.match(des1,des2,mask=mask)
    
    return matches_cross

def draw_arrows(vis_orig, points1, points2, color = (0, 255, 0), thick = 2, tip_length = 0.25):
    if len(vis_orig.shape) == 2: vis = cv2.cvtColor(vis_orig,cv2.COLOR_GRAY2RGB)
    else: vis = vis_orig
    for p1,p2 in zip(points1,points2):
        cv2.arrowedLine(vis, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), 
                        color=color, thickness=thick, tipLength = tip_length)
    return vis


def draw_point_tracks(kp1,img_next,kp2, bool_mask=None, display_invalid=False, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''
    #bool_mask = mask[:,0].astype(bool)
    if bool_mask is None:
        valid_left_matches = kp1
        valid_right_matches = kp2
    else:
        valid_left_matches = kp1[bool_mask,:]
        valid_right_matches = kp2[bool_mask,:]
    #img_right_out = draw_points(img_right, valid_right_matches)
    img_right_out = draw_arrows(img_next, valid_left_matches, valid_right_matches, 
                                color=color, thick = round(img_next.shape[1]/300))
    
    return img_right_out


def knn_match_and_lowe_ratio_filter(matcher, des1, des2,threshold=0.9, dist_mask_12=None):
    # First match 2 against 1
    if dist_mask_12 is None:
        dist_mask_21 = None
    else:
        dist_mask_21 = dist_mask_12.T
    matches_knn = matcher.knnMatch(des2,des1, k=2, mask = dist_mask_21 )
    #print("Len of knn matches", len(matches_knn))
    matches = []
    # Run lowes filter and filter with difference higher than threshold this might
    # still leave multiple matches into 1 (train descriptors)
    # Create mask of size des1 x des2 for permissible matches
    mask = np.zeros((des1.shape[0],des2.shape[0]),dtype='uint8')
    for match in matches_knn:
        if len(match)==1 or (len(match)>1 and match[0].distance < threshold*match[1].distance):
           # if match[0].distance < 75:
                matches.append(match[0])
                mask[match[0].trainIdx,match[0].queryIdx] = 1
    # run matches again using mask but from 1 to 2 which should remove duplicates            
    # This is basically same as running cross match after lowe ratio test
    matches_cross = matcher.match(des1,des2,mask=mask)
    
    return matches_cross

def read_matches_numpy(datapath, genC, matches_file, img1_name, img2_name):
    data = np.load(os.path.join(datapath,matches_file))
    src_matches = data[0]
    dst_matches = data[1]
    corrs1=np.zeros((6,0))
    corrs2=np.zeros((6,0))
    # this is boolean to make sure we have enough correspondences to send to recover pose
    done= False
    for i, pair in enumerate(src_matches):
        print(pair)
        c1_ind = i // 4 #num of cameras
        c2_ind = i % 4
        
        src_pts = src_matches[pair]
        dst_pts = dst_matches[pair] - [720,0]
        #get the undistorted normalized coordinates
        dst1 = cv2.undistortPoints(src_pts,genC.cams[c1_ind].K,genC.cams[c1_ind].dist_coeffs)
        dst2 = cv2.undistortPoints(dst_pts,genC.cams[c2_ind].K,genC.cams[c2_ind].dist_coeffs)
        
        #select the matches in one camera to pass over to recover pose.
        # we might not need it eventually with the full implementation of the
        # GEC code
        if c1_ind == c2_ind and not done:
            corrs1_img = dst1[:,0,:].T
            corrs2_img = dst2[:,0,:].T
            if dst1.shape[0] > 10:
                done = True
        if c1_ind != c2_ind: 
            pl1 = genC.get_plucker_coords_cam(dst1[:,0,:].T, c1_ind, normalize = False)
            pl2 = genC.get_plucker_coords_cam(dst2[:,0,:].T, c2_ind, normalize = False)
            
            corrs1 = np.append(corrs1, pl1, axis=1)
            corrs2 = np.append(corrs2, pl2, axis=1)
    
            img1 = cv2.imread(os.path.join(datapath, pair[0],img1_name))
            img2 = cv2.imread(os.path.join(datapath, pair[1],img2_name))
            img5 = displayMatches_manual(img1,src_matches[pair],img2,dst_matches[pair])
            #ax.cla()
            #im = ax.imshow(img5)
            #plt.show()
            cv2.imshow("matches"+str(c1_ind)+","+str(c2_ind),img5)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return corrs1, corrs2, corrs1_img, corrs2_img

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf):
    """ Compute a bounding_box filter on the given points

    Parameters
    ----------                        
    points: (n,2) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1],
                ...,
                [xn,yn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be keept or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

    bb_filter = np.logical_and(bound_x, bound_y)

    return bb_filter


def tiled_features(kp, img_shape, tiles_hor, tiles_ver, no_features = None):
    '''
    Given a set of keypoints, this divides the image into a grid and returns 
    len(kp)/(tiles_ver*tiles_hor) maximum responses within each tell. If that cell doesn't 
    have enough points it will return all of them.
    '''
    if no_features:
        feat_per_cell = np.ceil(no_features/(tiles_ver*tiles_hor)).astype(int)
    else:
        feat_per_cell = np.ceil(len(kp)/(tiles_ver*tiles_hor)).astype(int)
    HEIGHT, WIDTH = img_shape
    assert WIDTH%tiles_hor == 0, "Width is not a multiple of tiles_ver"
    assert HEIGHT%tiles_ver == 0, "Height is not a multiple of tiles_hor"
    w_width = int(WIDTH/tiles_hor)
    w_height = int(HEIGHT/tiles_ver)
        
    kps = np.array([])
    #pts = np.array([keypoint.pt for keypoint in kp])
    pts = cv2.KeyPoint_convert(kp)
    kp = np.array(kp)
    
    #img_keypoints = draw_markers( cv2.cvtColor(raw_images[0], cv2.COLOR_GRAY2RGB), kp, color = ( 0, 255, 0 ))

    
    for ix in range(0,HEIGHT, w_height):
        for iy in range(0,WIDTH, w_width):
            inbox_mask = bounding_box(pts, iy, iy+w_height, ix, ix+w_height)
            inbox = kp[inbox_mask]
            inbox_sorted = sorted(inbox, key = lambda x:x.response, reverse = True)
            inbox_sorted_out = inbox_sorted[:feat_per_cell]
            kps = np.append(kps,inbox_sorted_out)
            
            #img_keypoints = draw_markers(img_keypoints, kps.tolist(), color = [255, 0, 0] )
            #cv2.imshow("Selected Keypoints", img_keypoints )
            #print("Size of Tiled Keypoints: " ,len(kps))
            #cv2.waitKey(); 
    return kps.tolist()
        

def compute_matches(data_path, genC, cam_dirs, images, img1_ind, img2_ind,des_type='orb'):
    #extract key points from both frames
    
    #create the keypoint detector
    if des_type == 'orb':
        kp_detector = cv2.ORB_create()
        matcher= cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    else:
        kp_detector = zernike.MultiHarrisZernike(Nfeats=100)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    frame1 = {}
    frame1_imgs=[]
    print("Reading Frame 1 data")
    for d in cam_dirs:
        cam_num = int(d[3:])
        img_path = os.path.join(data_path, d, images[img1_ind])
        img = cv2.imread(img_path,0)
        print(img_path)
        frame1_imgs.append(img)
        if des_type == 'orb':
            kp= kp_detector.detect(img,None)
            kp = tiled_features(kp, img.shape, 6, 4, no_features= 100)
            kp, des = kp_detector.compute(img, kp)
        else:
            kp, des = kp_detector.detectAndCompute(img,None)
        frame1[cam_num] = kp,des
    
    #Second camera frame
    print("Reading Frame 2 data")
    frame2 = {}
    frame2_imgs=[]
    for d in cam_dirs:
        cam_num = int(d[3:])
        img_path = os.path.join(data_path, d, images[img2_ind])
        img = cv2.imread(img_path,0)
        print(img_path)
        frame2_imgs.append(img)
        if des_type == 'orb':
            kp= kp_detector.detect(img,None)
            kp = tiled_features(kp, img.shape, 6, 4, no_features= 100)
            kp, des = kp_detector.compute(img, kp)
        else:
            kp, des = kp_detector.detectAndCompute(img,None)
        frame2[cam_num] = kp,des
    
    
    # find matching correspondences
       
    # find the inter camera correspondences. 
    #The brute force way is to match every camera key points 
    # in frame 1 with every other camera in frame 2 and save the matches
    
    corrs1=np.zeros((6,0))
    corrs2=np.zeros((6,0))
    cam1_inds = []
    cam2_inds=[]
    # this is boolean to make sure we have enough correspondences to send to recover pose
    done= False
    #for each camera in frame1
    for i,cam_no in enumerate(frame1):
        #extract key points and descriptors
        kp1, des1 = frame1[cam_no]
        img1= frame1_imgs[i]
        # for each camera in frame 2
        for j, cam_no2 in enumerate(frame2):
            #if j < i:
            #    continue
            #print(" Matching "+str(i) + "," + str(j))
            kp2, des2 = frame2[cam_no2]
            img2= frame2_imgs[j]
            
            #find the matches
            #matches = bf.match(des1,des2)
            matches = knn_match_and_lowe_ratio_filter(matcher, des1, des2, threshold=0.9)
            #sort the matches based on distance
            #matches = sorted(matches, key = lambda x:x.distance)
            #take only top 50 matches
            #matches = matches[:10]
            kp1_matched_pts = np.array([kp1[mat.queryIdx].pt for mat in matches])
            kp2_matched_pts = np.array([kp2[mat.trainIdx].pt for mat in matches])
            
            #get the undistorted normalized coordinates
            dst1 = cv2.undistortPoints(kp1_matched_pts,genC.cams[i].K,genC.cams[i].dist_coeffs)
            dst2 = cv2.undistortPoints(kp2_matched_pts,genC.cams[j].K,genC.cams[j].dist_coeffs)
            
            #select the matches in one camera to pass over to recover pose.
            # we might not need it eventually with the full implementation of the
            # GEC code
            if i == j and not done:
                corrs1_img = dst1[:,0,:].T
                corrs2_img = dst2[:,0,:].T
                if dst1.shape[0] > 10:
                    done = True
            if i != j: 
                pl1 = genC.get_plucker_coords_cam(dst1[:,0,:].T, i, normalize = False)
                pl2 = genC.get_plucker_coords_cam(dst2[:,0,:].T, j, normalize = False)
                print(" Matching "+str(i) + "," + str(j))
                corrs1 = np.append(corrs1, pl1, axis=1)
                corrs2 = np.append(corrs2, pl2, axis=1)
                cam1_inds = cam1_inds + [i]*dst1.shape[0]
                cam2_inds = cam2_inds + [j]*dst2.shape[0]
                #img5 = displayMatches(img1,kp1,img2,kp2,matches,None)
                
                #cv2.imshow("matches"+str(i)+","+str(j),img5)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
    return corrs1, corrs2, corrs1_img, corrs2_img, cam1_inds, cam2_inds

def normalize_corrs(corrs1, corrs2):
    pass

from helper_functions import ransac
def triangulate_linear(R, t, u1, u2):
    #here u1 and u2 are the normalized rays 
    #in the first and second body frames of the 
    #light field camera. They are already compensated
    #for the individual camera rotations
    
    P1 = np.append(np.eye(3), np.zeros((3,1)), axis=1)
    P2 = np.append(R,t[:, np.newaxis], axis=1)
    A = np.zeros((4,4))
    A[0, :] = u1[0] * P1[2, :] - u1[2] * P1[0, :]
    A[1, :] = u1[1] * P1[2, :] - u1[2] * P1[1, :]
    A[2, :] = u2[0] * P2[2, :] - u2[2] * P2[0, :]
    A[3, :] = u2[1] * P2[2, :] - u2[2] * P2[1, :]
    
    U, D, Vh = np.linalg.svd(A)
    pt_3D = Vh[3, :-1]
    pt_3D = pt_3D / Vh[3, -1]
    return pt_3D

def sanity_check(best_inliers, best_T, corrs1, corrs2):
    pts = np.zeros((3, corrs1.shape[1]))
    for i in best_inliers:
        pt = triangulate_linear(best_T[0:3,0:3], best_T[:,3], corrs1[0:3, i], corrs2[0:3, i])
        pts[:,i] = pt
        re_proj1 = pt / np.linalg.norm(pt)
        pt = helper.euclid_to_homo(pt[:, np.newaxis])
        re_proj2 = best_T @ pt
        re_proj2 = re_proj2 / np.linalg.norm(re_proj2)
        print("-------"+str(i)+"--------")
        print("original : \n")
        print (corrs1[0:3, i])
        print (corrs2[0:3, i])
        print("reprojected : \n")
        print(re_proj1)
        print(re_proj2)
        
        #err1 = 1.0 - np.dot(u1 , re_proj1)
        #err2 = 1.0 - np.dot(u2 , re_proj2)
    # reproject the 3D points back and  see if they match with correspondences
    
    
def ransac_estimate(ax4, corrs1, corrs2,corrs1_img, corrs2_img, cam_inds1, cam_inds2,imgs, genC, gec_solver):
    #create the ransac object
    ransac_obj = ransac.Ransac()
    ransac_obj.threshold = 2.0*(1.0 - np.cos(np.arctan(1.0/800.0)))
    ransac_obj.max_iterations = 500
    # initialize the parameters
    num_best_inliers =0
    best_inliers = []
    best_T = np.zeros((3,4))
    iteration = 0
    
    
    #fig4, ax4 = initialize_2d_plot(number=4)
    
    while iteration < ransac_obj.Num_trials :
        #generate the random samples. choose randomly
        #minimum number of samples required to estimate the essential matrix
        rand_inds = np.random.choice(range(corrs1.shape[1]),ransac_obj.sample_size,replace=False)
        sample_corrs1= corrs1[:,rand_inds]
        sample_corrs2 = corrs2[:,rand_inds]
        
        #compute the model = essential matrix
        R_final, t_final = gec_solver.solve_gec(sample_corrs1, sample_corrs2, corrs1_img, corrs2_img)
        inlier_count =0
        inliers = []
        #count the number of inliers
        #calculate the distance function to the model
        # we are following the technique similar to openGV
        # trainagulate the correspondence and reproject the 
        #point and calculate if the reprokection if within a threshold cone
        P1 = np.append(np.eye(3), np.zeros((3,1)), axis=1)
        P2 = np.append(R_final , t_final[:, np.newaxis], axis=1)
        
        cnt_neg_depth=0
        for i in range(corrs1.shape[1]):
            u1 = corrs1[0:3, i]
            u2 = corrs2[0:3, i]
            pt_3D = triangulate_linear(R_final, t_final, u1, u2)
            #print("\npt 3D _1 :")
            #print(pt_3D)
            #for now bring the  orrespondences back into camera frame
            #now u1 and u2 are in the camera coordinate system
            cam1 = cam_inds1[i]
            cam2 = cam_inds2[i]
            u1 = genC.cams[cam1].poseR.T @ u1
            u2 = genC.cams[cam2].poseR.T @ u2
            #get the pose of the curreny GenCam
            TT = helper.compose_T(R_final, t_final[np.newaxis,:])
            TT = helper.T_inv(TT)
            TT = TT[:-1, :]
            
            dir_trans =  genC.cams[cam1].poseR.T @ ((TT[:,3:4] - genC.cams[cam1].poset) + TT[0:3,0:3] @ genC.cams[cam2].poset)
            dir_rot = genC.cams[cam1].poseR.T * TT[0:3,0:3] * genC.cams[cam2].poseR
            P2_t =  -dir_rot.T @ dir_trans;
            P2 = np.append(dir_rot.T , P2_t , axis=1)
            pt_3D = triangulate_linear(dir_rot.T, P2_t[:,0], u1, u2)
            #print("\npt 3D _2 :")
            #print(pt_3D)
            if pt_3D[2] < 0 :
                cnt_neg_depth = cnt_neg_depth+1
            re_proj1 = pt_3D / np.linalg.norm(pt_3D)
            pt_3D = helper.euclid_to_homo(pt_3D[:, np.newaxis])
            re_proj2 = P2 @ pt_3D
            re_proj2 = re_proj2 / np.linalg.norm(re_proj2)
            err1 = 1.0 - np.dot(u1 , re_proj1)
            err2 = 1.0 - np.dot(u2 , re_proj2)
            err = (err1 + err2)
            if err <  ransac_obj.threshold :
                inlier_count = inlier_count+1
                inliers.append(i)
        #print("inliers cnt : "+str(inlier_count))
        #print(" inliers: ")
        #print(inliers)
        #TT = helper.compose_T(R_final, t_final[np.newaxis,:])
        #print(TT)
        #print("---------------------------------------")
        #rand_inds.sort()
        #print(rand_inds)
        if inlier_count > num_best_inliers :
            num_best_inliers = inlier_count
            best_T = np.append(R_final, t_final[:, np.newaxis], axis=1)
            best_inliers = inliers
            #adaptively compute the numer of trials
            w = num_best_inliers/ corrs1.shape[1]
            prob_outliers = 1- np.power(w , len(rand_inds) )
            prob_outliers = min(max(prob_outliers,   np.finfo(float).eps), 1.0 - np.finfo(float).eps) 
            ransac_obj.Num_trials = np.log(1- ransac_obj.prob) /np.log(prob_outliers)
            print("iteration: "+str(iteration)+", num_best_inliers : "+str(num_best_inliers)\
                  +" inlier ratio : "+str(w)+" num_trials : "+str(ransac_obj.Num_trials))
            print(best_T)
            #pts_3Dcv = cv2.triangulatePoints(P1, P2, corrs1[0:2, :], corrs2[0:2, :])
            print("---------------------------------------")
            #visualize the feature tracks and inliers
            #define colors
            colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255)]
            inds1 = cam_inds1[best_inliers]
            inds2 = cam_inds2[best_inliers]
            pts1 =  corrs1[0:3, best_inliers]
            pts2 =  corrs2[0:3, best_inliers]
            for i,img in enumerate(imgs):
                draw_img = img
                #Extract the indices of all the matches
                # of image i in the current frame and draw the colored matches  
                mask_i_cam = (inds2==i)                            
                pts2_tmp = genC.cams[i].K @ pts2[:, mask_i_cam]
                pts2_tmp = helper.homo_to_euclid(pts2_tmp)
                
                cam1_matched_inds = inds1[mask_i_cam]
                pts1_tmp = pts1[:, mask_i_cam]
                for j in range(len(imgs)):
                    mask_j_cam = (cam1_matched_inds == j)
                    pts1_tmp_tmp = genC.cams[j].K @ pts1_tmp[:,mask_j_cam]
                    pts1_tmp_tmp = helper.homo_to_euclid(pts1_tmp_tmp)
                    pts2_tmp_tmp = pts2_tmp[:,mask_j_cam]
                    #inds1==j
                    draw_img = draw_point_tracks(pts1_tmp_tmp.T, draw_img, pts2_tmp_tmp.T, None, True, color=colors[j])
                
                ax4[i//3,i%3].imshow(draw_img)
                #ax4.imshow(draw_img)
            plt.show()
            plt.pause(.5)
                #cv2.imshow("matches"+str(i),draw_img)
                
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
            
        iteration = iteration + 1
        if iteration > ransac_obj.max_iterations :
            print("ransac reached maximum iterations")
            break
    return best_T , best_inliers
        
        
# paths of the calibration files and the data folder
calib_file = "data/camchain_isec_2020.yaml"    #"data/camchain-aug_2019.yaml"
data_path= "data/curry_back_2020"

#Initialize plots
fig3,ax3 = helper.initialize_3d_plot(number=3, limits = np.array([[-2,2],[-2,2],[-5,5]]), view=[-30,-90])


# create a generalized camera from the info
genC = genCam.GeneralizedCamera()
genC.build_camera_system(calib_file)
# This camera has five cameras we need only four 0,2,3,4 indices
# so remove the index 1 camera manually
#del genC.cams[1]
#genC.num_cams = 4
genC.plot_camera(ax3, 12, 0, color="orange")
gec_solver = gec.GECSolver()

# read the images from respective folders and extract key points
cam_dirs=[]

for d in os.listdir(data_path):
    if 'cam' in d and d[3:].isdigit():
        cam_num = int(d[3:])
        cam_dirs.append(d)
cam_dirs.sort()

#We trust that all 
#directories has same number of synced images with matching image names
# retrive the image list from 1st directory and use it to read images
# form other cameras 
images = os.listdir(os.path.join(data_path, cam_dirs[0]))
images.sort()
#images.reverse()
#images = images[-40 : ]
ims = [ im*5   for im in range(len(images)//5)]
#print(images)

fig4, ax4 = helper.initialize_2d_plot_multi(number=4,num_rows=2, num_cols=3)
fig4.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.1, hspace=0.11)
fig4.set_size_inches((10,8))
[axi.set_axis_off() for axi in ax4[:2,:].ravel()]
for ind, im in  enumerate(ims):
    if ind+1 >= len(ims):
        continue
    corrs1, corrs2, corrs1_img, corrs2_img, cam_inds1, cam_inds2 = compute_matches(data_path, genC, cam_dirs, images, im, ims[ind+1], 'orb')
    #corrs1, corrs2, corrs1_img, corrs2_img = read_matches_numpy(data_path, genC, 'matches_new.npy', "1565655135.jpg" , "1565655137.jpg")
    #create a GEC solver and solve for the relative pose estimation
    
    #normalize the correspondences
    
    #R_final, t_final = gec_solver.solve_gec(corrs1, corrs2, corrs1_img, corrs2_img)
    #TT =helper.compose_T(R_final, t_final[np.newaxis,:])
    #TT = helper.T_inv(TT)
    #print(TT)
    
    #ransac estimation
    frame2_imgs=[]
    for i,d in enumerate(cam_dirs):
        img_path = os.path.join(data_path, d, images[ims[ind+1]])
        img = cv2.imread(img_path,0)
        img = cv2.undistort(img, genC.cams[i].K,genC.cams[i].dist_coeffs)
        frame2_imgs.append(img)

    T_final, inliers = ransac_estimate(ax4, corrs1, corrs2,corrs1_img, corrs2_img, np.array(cam_inds1), np.array(cam_inds2),frame2_imgs, genC , gec_solver)
    TT = helper.T_inv(np.append(T_final, [[0,0,0,1]], axis=0))
    
    genC.transform(TT[0:3, 0:3], TT[0:3, 3:4])
    genC.plot_camera(ax3,12, 0, color="magenta") 
    plt.pause(.25)
    #plt.show(block=False)
    #do find essential for a single camera to see what we get
    # c_ind=0;
    # kp1, des1 = frame1[c_ind]
    # img1= frame1_imgs[c_ind]
    # kp2, des2 = frame2[c_ind]
    # img2= frame2_imgs[c_ind]
            
    # #find the matches
    # matches = bf.match(des1,des2)
    
    # kp1_matched_pts = np.array([kp1[mat.queryIdx].pt for mat in matches])
    # kp2_matched_pts = np.array([kp2[mat.trainIdx].pt for mat in matches])
    
    # #get the undistorted normalized coordinates
    # dst1 = cv2.undistortPoints(kp1_matched_pts,genC.cams[c_ind].K,genC.cams[c_ind].dist_coeffs)
    # dst2 = cv2.undistortPoints(kp2_matched_pts,genC.cams[c_ind].K,genC.cams[c_ind].dist_coeffs)

    E, mask = cv2.findEssentialMat(corrs1_img.T, corrs2_img.T, focal=1.0, pp=(0., 0.), 
                               method=cv2.RANSAC, prob=0.999, threshold=0.001)
 
    print ("Essential matrix used ",np.sum(mask) ," of total ", corrs1_img.shape[1],"matches")

    #img5 = displayMatches(img1,kp1,img2,kp2,matches,mask)

    points, R, t, mask_recPose = cv2.recoverPose(E, corrs1_img.T, corrs2_img.T,mask=mask)
    print("points:",points)
    print("R:",R)
    print("t:",t.transpose())
    
    
    TT =helper.compose_T(R, t.T)
    TT = helper.T_inv(TT)
    print(TT)
    # create a generalized camera from the info
    genC2 = genCam.GeneralizedCamera()
    genC2.build_camera_system(calib_file)
    genC2.transform(TT[0:3, 0:3], TT[0:3, 3:4])
    genC2.plot_camera(ax3,12, 0,color="blue")
    plt.pause(.25)
    
################ Normali essential matrix give right answer ##############