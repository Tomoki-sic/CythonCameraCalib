from concurrent.futures import thread
from itertools import filterfalse
from turtle import circle
import cv2
import numpy as np
import sys

sys.path.append("C:\\Users\\Tomoki\\source\\repos\\CythonCameraCalibration\\CythonCameraCalibration\\src\\dll")

from dll import setting
from dll import flyCapture
from dll import imageProcessing
from dll import cameraCalibration as cal

CAMERA = "STSD2_002"
K = "./"+CAMERA+"/cal9/K.pkl"
DIST = "./"+CAMERA+"/cal9/dist.pkl"

OUTPUT = "./"+CAMERA+"/circleGrid6/"

ZHANG_METHOD = False
PHOTOGRAPHY = True
PICKUP = False
INTETPOLATION = False
FUSIONPOINT = False
CALIBFROMLOOK = False
IMAGE_WIDTH = 2080
IMAGE_HEIGHT = 1552
DST_WIDTH = 2080
DST_HEIGHT = 1552
SHAPE = (10,9)
F = 0.25

def nothing(val):
    pass

set = setting.setPram(CAMERA,K,DIST)

args = sys.argv
mtx = set.getMtx()
dist = set.getDist()
file_num = args[1]
threshold = 80
rect_min=3
rect_max=200

print(mtx,dist)

Syn_x = set.loadPickle(OUTPUT+"Synthesized_Table_x.binary")
Syn_y = set.loadPickle(OUTPUT+"Synthesized_Table_y.binary")


cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
cv2.namedWindow("capture_pre", cv2.WINDOW_NORMAL)
img_undist =  np.ones((DST_HEIGHT,DST_WIDTH,3),np.uint8) * 0
cv2.createTrackbar("threshold","capture",threshold,255,nothing)
cv2.createTrackbar("rect_min","capture",rect_min,50,nothing)
cv2.createTrackbar("rect_max","capture",rect_max,300,nothing)

if PHOTOGRAPHY:
    cap = flyCapture.pointGreyCamera2OpenCV(IMAGE_WIDTH,IMAGE_HEIGHT)
    cap.startCapture()
    while True:
        k = cv2.waitKey(1) & 0xFF
        reg, img_dist = cap.captureImage()
        threshold = cv2.getTrackbarPos("threshold","capture")
        rect_min = cv2.getTrackbarPos("rect_min","capture")
        rect_max = cv2.getTrackbarPos("rect_max","capture")
        if ZHANG_METHOD:    
            img_undist = cal.cameraCalibrateFromKandDist(img_dist,mtx,dist,(DST_HEIGHT,DST_WIDTH))
            circles_pre = cal.labeling(img_dist, threshold=threshold,show=False)
            img_pre = cal.drawCircles(img_dist,circles_pre)
            cv2.imshow("capture_pre",img_pre)
            circles = cal.labeling(img_undist,threshold=threshold)
            img2 = cal.drawCircles(img_undist,circles)
            cv2.imshow("capture",img2)
        else:
            #img_hsv = cv2.cvtColor(img_dist,cv2.COLOR_RGB2HSV)
            img_undist = cal.cameraCalibrateFromLookUpTable(img_dist,Syn_x, Syn_y,DST_WIDTH,DST_HEIGHT)
            img_dist2 = cv2.resize(img_dist, dsize = None,  fx=F,fy = F)
            circles_pre = cal.labeling(img_dist2,threshold=threshold, rect_min=rect_min,rect_max=rect_max,show=False)
            img_pre = cal.drawCircles(img_dist2,circles_pre)
            cv2.imshow("capture",img_pre[:,:,1])
            cv2.imshow("capture_pre",img_undist)
        if k == 27:
            cap.stopCapture()
            break
        elif k == ord('a'):
            img_undist = cv2.resize(img_undist, dsize = None,  fx=F,fy = F)
            cv2.imwrite(OUTPUT+file_num+"_undist.bmp",img_undist)
            cv2.imwrite(OUTPUT+file_num+"_dist.bmp",img_dist)
            cv2.waitKey(1)
            break

cv2.destroyAllWindows()
if PICKUP:
    img_undist = cv2.imread(OUTPUT+file_num+"_undist.bmp")
    img_dist = cv2.imread(OUTPUT+file_num+"_dist.bmp")
    img_dist = cv2.resize(img_dist, dsize = None,  fx=F,fy = F)
    points_undist, circleGrid = cal.calculateUndistortedPoint(img_undist,(3,3),SHAPE,threshold)
    points_dist = cal.pickUpDistortedPoints(img_dist,threshold,F)        
    p1, p2 = cal.pickUpCirclesAndSort(img_undist,points_undist,points_dist,OUTPUT+file_num)
    set.saveLookUpTable(OUTPUT+file_num+".csv",circleGrid,p1, p2)

if INTETPOLATION:    
    u_undist, v_undist, u_dist, v_dist = set.loadLookUpTable(OUTPUT+file_num+".csv")
    #dst_x,  dst_y = cal.InterpolateLookUpTable(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape=SHAPE)
    dst_x,  dst_y = cal.InterpolateLookUpTable(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape=SHAPE)
    set.writePickle(OUTPUT+"x/"+file_num+"_complemented_x.binary",dst_x)
    set.writePickle(OUTPUT+"y/"+file_num+"_complemented_y.binary",dst_y)
    set.saveComplementedImage(OUTPUT+file_num, DST_WIDTH, DST_HEIGHT, dst_x, dst_y)

if FUSIONPOINT:
    dst_x, dst_y = cal.SynthesizeTable(OUTPUT+"x/",OUTPUT+"y/",DST_WIDTH,DST_HEIGHT,set)
    set.writePickle(OUTPUT+"Synthesized_Table_x.binary",dst_x)
    set.writePickle(OUTPUT+"Synthesized_Table_y.binary",dst_y)

if CALIBFROMLOOK:
    test = cv2.imread(OUTPUT+"test.bmp")
    Syn_x = set.loadPickle(OUTPUT+"Synthesized_Table_x.binary")
    Syn_y = set.loadPickle(OUTPUT+"Synthesized_Table_y.binary")
    dst = cal.cameraCalibrateFromLookUpTable(test,Syn_x, Syn_y,DST_WIDTH,DST_HEIGHT)
    cv2.imwrite(OUTPUT+file_num+"_dst.bmp",dst)
    set.saveComplementedImage(OUTPUT+"whole", DST_WIDTH, DST_HEIGHT, Syn_x, Syn_y)