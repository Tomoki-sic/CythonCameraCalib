import wafSIFT
import  cv2
import numpy as np
import os
import plotly.offline as po
import plotly.graph_objs as go
import pandas as pd
import random

def drawLine(img1, img2,pts1,pts2,scale = 1,thick = 1,rad = 5):
    h, w = img1.shape[:1]
    img = cv2.hconcat([img1, img2])
    for i in range(len(pts1)):
        x1,y1 = int(pts1[i][0]),int(pts1[i][1])
        x2,y2 = int(pts2[i][0]+w/scale),int(pts2[i][1])
        a = random.randint(0,255)
        b = random.randint(0,255)
        c = random.randint(0,255)
        img = cv2.circle(img, (x1,y1), rad, (a,b,c),-1)
        img = cv2.circle(img, (x2,y2), rad, (a,b,c),-1)
        img = cv2.line(img, (x1,y1), (x2,y2), (a, b, c),thickness=thick)
    return img


input_dir = "C:\\Users\\Tomoki\\Desktop\\src\\input6"
output_dir = "C:\\Users\\Tomoki\\Desktop\\wafSLAM\\output3"

CAMERA = "STSD2_002"
K = "./"+CAMERA+"/calib_zhang/K.pkl"
DIST = "./"+CAMERA+"/calib_zhang/dist.pkl"
OUTPUT = "./"+CAMERA+"/calib3/"


#image
img_name1 = "c"
img_name2 = "yoko"

#WAFセンサ用パラメータ
cx = 1040
cy = 776
fx = 0
fy = 0
th = 0.75
K = np.array([[fx, 0.000000000000000000e+00, cx],[0.000000000000000000e+00, fy, cy],[0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
dist_coef = np.array([0, 0, 0, 0, 0])

img1 = cv2.imread(OUTPUT+img_name1)
img2 = cv2.imread(OUTPUT+img_name2)
height, width = img1.shape[:1]
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.05)
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
for m,n in matches:
    if m.distance < th*n.distance:
        good.append([m])
        pts2.append((kp2[m.trainIdx].pt[0],kp2[m.trainIdx].pt[1]))
        pts1.append((kp1[m.queryIdx].pt[0],kp1[m.queryIdx].pt[1]))


match = sift.drawLine(img_con.copy(),pts1,pts2,thick=3,rad=7)
cv2.imwrite("match.bmp",match)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K, distCoeffs=None)
pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K, distCoeffs=None)
E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
points, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
print(R,t)






