from concurrent.futures import thread
from itertools import filterfalse
from turtle import circle
import cv2
import numpy as np
import sys

from dll import setting
from dll import flyCapture
from dll import imageProcessing
from dll import cameraCalibration as cal

CAMERA = "STSD2_002"
K = "./"+CAMERA+"/calib_zhang/K.pkl"
DIST = "./"+CAMERA+"/calib_zhang/dist.pkl"

OUTPUT = "./"+CAMERA+"/calib_zhang/"

IMAGE_WIDTH = 2080
IMAGE_HEIGHT = 1552
DST_WIDTH = 2080
DST_HEIGHT = 1552
SHAPE = (10,9)
F = 0.5

set = setting.setPram(CAMERA,K,DIST)

args = sys.argv
mtx = set.getMtx()
dist = set.getDist()
file_name = args[1]

img = cv2.imread(OUTPUT+file_name)
Syn_x = set.loadPickle(OUTPUT+"Synthesized_Table_x2.binary")
Syn_y = set.loadPickle(OUTPUT+"Synthesized_Table_y2.binary")

#img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    

#img_undist1 = cal.cameraCalibrateFromKandDist(img,mtx,dist,(DST_HEIGHT,DST_WIDTH))
img_undist2 = cal.cameraCalibrateFromLookUpTable(img,Syn_x, Syn_y,DST_WIDTH,DST_HEIGHT)

img_re = cv2.resize(img_undist2, dsize = None,  fx=F,fy = F)
            
cv2.imwrite(OUTPUT+"img_undist2.bmp",img_undist2)
cv2.imwrite(OUTPUT+"img_undist3.bmp",img_re)