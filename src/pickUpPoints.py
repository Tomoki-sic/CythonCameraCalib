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

CAMERA = "STSD3_002"
K = "./"+CAMERA+"/calib_zhang2/K.pkl"
DIST = "./"+CAMERA+"/calib_zhang2/dist.pkl"

OUTPUT = "./"+CAMERA+"/6/"

ZHANG_METHOD = False
PHOTOGRAPHY = True
PICKUP = True   
INTETPOLATION = True
FUSIONPOINT = True
CALIBFROMLOOK = True
IMAGE_WIDTH = 2080
IMAGE_HEIGHT = 1552
SHAPE = (6,8)
F = 1
DST_WIDTH = int(2080*F)
DST_HEIGHT = int(1552*F)

def nothing(val):
    pass
set = setting.setPram(CAMERA,K,DIST)


args = sys.argv
mtx = set.getMtx()
dist = set.getDist()
file_num = args[1]
threshold = 80
threshold2 = 80
rect_min=3
rect_max=400

print(mtx,dist)

Syn_x = set.loadPickle(OUTPUT+"Synthesized_Table_x6.binary")
Syn_y = set.loadPickle(OUTPUT+"Synthesized_Table_y6.binary")


cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
cv2.namedWindow("capture_pre", cv2.WINDOW_NORMAL)
img_undist =  np.ones((DST_HEIGHT,DST_WIDTH,3),np.uint8) * 0
cv2.createTrackbar("threshold","capture",threshold,255,nothing)
cv2.createTrackbar("threshold2","capture_pre",threshold2,255,nothing)

cv2.createTrackbar("rect_min","capture",rect_min,50,nothing)
cv2.createTrackbar("rect_max","capture",rect_max,300,nothing)

if PHOTOGRAPHY:
    cap = flyCapture.pointGreyCamera2OpenCV(IMAGE_WIDTH,IMAGE_HEIGHT)
    cap.startCapture()
    while True:
        k = cv2.waitKey(1) & 0xFF
        reg, img_dist = cap.captureImage()
        threshold = cv2.getTrackbarPos("threshold","capture")
        threshold2 = cv2.getTrackbarPos("threshold2","capture_pre")
        rect_min = cv2.getTrackbarPos("rect_min","capture")
        rect_max = cv2.getTrackbarPos("rect_max","capture")
        if ZHANG_METHOD:    
            img_undist = cal.cameraCalibrateFromKandDist(img_dist,mtx,dist,(DST_HEIGHT,DST_WIDTH),F=F)
            circles_pre = cal.labeling(img_dist, threshold=threshold,show=True)
            img_pre = cal.drawCircles(img_dist,circles_pre)
            cv2.imshow("capture_pre",img_pre)
            circles = cal.labeling(img_undist,threshold=threshold,show=True)
            img2 = cal.drawCircles(img_undist,circles)
            cv2.imshow("capture",img2)
        else:
            img_undist = cal.cameraCalibrateFromLookUpTable(img_dist,Syn_x, Syn_y,DST_WIDTH,DST_HEIGHT)
            img_dist2 = cv2.resize(img_dist, dsize = None,  fx=F,fy = F)
            circles_pre = cal.labeling(img_dist,threshold=threshold, rect_min=rect_min,rect_max=rect_max,show=False)
            img_pre = cal.drawCircles(img_dist,circles_pre)
            circles = cal.labeling(img_undist,threshold=threshold2, rect_min=rect_min,rect_max=rect_max,show=False)
            img_pre_undist = cal.drawCircles(img_undist,circles)
            cv2.imshow("capture",img_pre[:,:,1])
            cv2.imshow("capture_pre",img_pre_undist[:,:,1])
        if k == 27:
            cap.stopCapture()   
            break
        elif k == ord('a'):
            cv2.imwrite(OUTPUT+file_num+"_undist.bmp",img_undist[:,:,1])
            cv2.imwrite(OUTPUT+file_num+"_dist.png",img_dist)
            cv2.imwrite(OUTPUT+file_num+"_dist2.png",img_dist[:,:,1])

            cv2.waitKey(1)
            break

cv2.destroyAllWindows()
if PICKUP:
    img_undist = cv2.imread(OUTPUT+file_num+"_undist.bmp")
    img_dist = cv2.imread(OUTPUT+file_num+"_dist.png")
    #img_dist = cv2.resize(img_dist, dsize = None,  fx=F,fy = F)
    points_undist, circleGrid = cal.calculateUndistortedPoint(img_undist,(6,8),SHAPE,threshold2)
    points_dist = cal.pickUpDistortedPoints(img_dist,threshold,1,rect_min,rect_max)        
    p1, p2 = cal.pickUpCirclesAndSort2(img_undist,points_undist,points_dist,OUTPUT+file_num,SHAPE)
    set.saveLookUpTable(OUTPUT+file_num+".csv",circleGrid,p1, p2)

if INTETPOLATION:    
    u_undist, v_undist, u_dist, v_dist = set.loadLookUpTable(OUTPUT+file_num+".csv")
    #dst_x,  dst_y = cal.InterpolateLookUpTable(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape=SHAPE)
    #dst_x,  dst_y = cal.InterpolateLookUpTable(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape=SHAPE)
    dst_x, dst_y = cal.InterpolateLookUpTable2(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape=SHAPE)
    set.writePickle(OUTPUT+"x/"+file_num+"_complemented_x.binary",dst_x)
    set.writePickle(OUTPUT+"y/"+file_num+"_complemented_y.binary",dst_y)
    set.saveComplementedImage(OUTPUT+file_num, DST_WIDTH, DST_HEIGHT, dst_x, dst_y)

if FUSIONPOINT:
    dst_x, dst_y = cal.SynthesizeTable(OUTPUT+"x/",OUTPUT+"y/",DST_WIDTH,DST_HEIGHT,set)
    set.writePickle(OUTPUT+"Synthesized_Table_x.binary",dst_x)
    set.writePickle(OUTPUT+"Synthesized_Table_y.binary",dst_y)

if CALIBFROMLOOK:
    test = cv2.imread(OUTPUT+"test.png")
    Syn_x = set.loadPickle(OUTPUT+"Synthesized_Table_x.binary")
    Syn_y = set.loadPickle(OUTPUT+"Synthesized_Table_y.binary")
    dst = cal.cameraCalibrateFromLookUpTable(test,Syn_x, Syn_y,DST_WIDTH,DST_HEIGHT)
    cv2.imwrite(OUTPUT+file_num+"_dst.bmp",dst)
    set.saveComplementedImage(OUTPUT+"whole", DST_WIDTH, DST_HEIGHT, Syn_x, Syn_y)