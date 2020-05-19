# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import os

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
matches_src = {}
matches_dst = {}
c1 = ""
c2 = ""
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, matches_src, matches_dst, c1,c2
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        matches_src[(c1,c2)] = np.append( matches_src[(c1,c2)] , np.array([refPt[0]]) , axis=0)
        matches_dst[(c1,c2)] = np.append( matches_dst[(c1,c2)] , np.array([refPt[1]]), axis=0)
        # draw a rectangle around the region of interes
        cv2.line(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True, help="Path to the image")
ap.add_argument("-i2", "--image2", required=True, help="Path to the image")
ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
ap.add_argument("-o", "--output",  default="matches.npy", help="Path to the yaml file")

args = vars(ap.parse_args())

# read the images from respective folders and extract key points
cam_dirs=[]
data_path = args["dir"]
for d in os.listdir(data_path):
    if 'cam' in d and d[3:].isdigit():
        cam_num = int(d[3:])
        cam_dirs.append(d)
cam_dirs.sort()


print("Reading Frame 1 data")

for d1 in cam_dirs:
    img_path = os.path.join(data_path, d1, args["image1"])
    image1 = cv2.imread(img_path,0)
    print("-----------------")
    for d2 in cam_dirs:
        img_path = os.path.join(data_path, d2, args["image2"])
        image2 = cv2.imread(img_path,0)
        print(d1+ ",,," + d2)
        c1=d1
        c2=d2
        matches_src[(c1,c2)] = np.zeros((0,2))
        matches_dst[(c1,c2)] = np.zeros((0,2))
        image = np.concatenate((image1, image2), axis=1)

        clone = image.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)
        # keep looping until the 'q' key is pressed
        while True: 
            # display the image and wait for a keypress
	        cv2.imshow("image", image)
	        key = cv2.waitKey(1) & 0xFF
            #if the 'r' key is pressed, reset the cropping region
	        if key == ord("r"):
		        image = clone.copy()
	        # if the 'c' key is pressed, break from the loop
	        elif key == ord("c"):
		        break
        cv2.destroyAllWindows()
# write all the matches into a yaml files
matches_file = os.path.join(data_path , args["output"])
#with open(yaml_file, 'w') as file:
np.save(matches_file , [matches_src, matches_dst])
# close all open windows
cv2.destroyAllWindows()
