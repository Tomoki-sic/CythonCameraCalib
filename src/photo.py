import cv2
import sys
import numpy as np

from dll import flyCapture
from dll import cameraCalibration as cal

CAMERA = "STSD2_002"
OUTPUT = "./"+CAMERA+"/cal9/"
IMAGE_WIDTH = 2080
IMAGE_HEIGHT = 1552


args = sys.argv
file_num = args[1]

cv2.namedWindow("capture", cv2.WINDOW_NORMAL)

img_undist =  np.ones((IMAGE_HEIGHT,IMAGE_WIDTH,3),np.uint8) * 0
cap = flyCapture.pointGreyCamera2OpenCV(IMAGE_WIDTH,IMAGE_HEIGHT)
cap.startCapture()

while True:
    k = cv2.waitKey(1) & 0xFF
    reg, img = cap.captureImage()
    img2  = cv2.circle(img.copy(), (1040,776),500,(255,255,255),thickness=0)
    cv2.imshow("capture",img2)
    if k == 27:
        cap.stopCapture()
        break
    elif k == ord('a'):
        cv2.imwrite(OUTPUT+file_num+"_undist.bmp",img)
        break

cv2.destroyAllWindows()
