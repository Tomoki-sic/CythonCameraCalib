import numpy as np
import cv2
import glob
import os
import pickle

from dll import setting
from dll import cameraCalibration as cal

def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)

def transrated_points(points):
  dst = np.array([[points[0]]])
  points.pop(0)
  for point in points:
    dst = np.append(dst,[[point]],axis=0)
  return dst


MANUAL_FLAG = False
CAMERA = "STSD3_002"
OUTPUT = "./"+CAMERA+"/calib_zhang2/"
INPUT = "./"+CAMERA+"/"

SHAPE = (7,8)
DISRANCE_OF_GRID =  16

threshold = 80
rect_min=3
rect_max=200
F = 0.5

def writePickle(filename, data):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
        f.close()


Board_x, Board_y = SHAPE[0], SHAPE[1]
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, DISRANCE_OF_GRID, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((Board_x*Board_y,3), np.float32)
objp[:,:2] = np.mgrid[0:Board_y,0:Board_x].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(OUTPUT+'*.bmp')

print(images)
i = 0
for fname in images:
    i+=1
    img = cv2.imread(fname)
    kernel = make_sharp_kernel(1)
    #img = cv2.filter2D(img, -1, kernel).astype("uint8")
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,2]
    gray = img[:,:,1]
    #gray[gray<60] = 0
    #gray[gray>=60] = 255
    #gray = cv2.medianBlur(gray, ksize=3)
    cv2.imwrite(OUTPUT+'img'+str(i)+'.png',gray)
    # Find the chess board corners
    if MANUAL_FLAG:
      ret = True
      points = cal.pickUpDistortedPoints(img,threshold,F,rect_min,rect_max)
      corners = transrated_points(points)
      # If found, add object points, image points (after refining them)
      if ret == True:
          objpoints.append(objp)
          corners2 = corners
          imgpoints.append(corners2)
    #ret, corners = cv2.findChessboardCorners(gray,(Board_y,Board_x))
    else:
      #ret, corners = cv2.findCirclesGrid(gray, (Board_y,Board_x),cv2.CALIB_CB_SYMMETRIC_GRID)
      ret, corners = cv2.findChessboardCorners(gray,(Board_y,Board_x))
      # If found, add object points, image points (after refining them)
      if ret == True:
          print(fname)
          objpoints.append(objp)
          corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
          imgpoints.append(corners2)
          # Draw and display the corners
          img = cv2.drawChessboardCorners(img, (Board_y,Board_x), corners2,ret)
          cv2.imwrite(OUTPUT+str(i)+".png",img)
    

    
#os.chdir("calib")
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None,flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print("???????????????:")
print(mtx)
print("????????????:")
print(dist)


writePickle(OUTPUT+"K.pkl",mtx)
writePickle(OUTPUT+"dist.pkl",dist)

img = cv2.imread(OUTPUT+'cal01_undist.bmp')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
block = np.ones((h,w,3),np.int32) * 0
print(newcameramtx)
print(ret)
x,y,w,h = roi
block[y:y+h, x:x+w] = dst[y:y+h, x:x+w]
#dst = dst[y:y+h, x:x+w]
cv2.imwrite(OUTPUT+'calibresult.png',block)
cv2.destroyAllWindows()