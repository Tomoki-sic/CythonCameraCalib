import cv2

CAMERA = "STSD2_002"
K = "./"+CAMERA+"/K.pkl"
DIST = "./"+CAMERA+"/dist.pkl"

OUTPUT = "./"+CAMERA+"/circleGrid5/"

img = cv2.imread(OUTPUT+"1_x.bmp")
cv2.namedWindow("test", cv2.WINDOW_NORMAL)

while True:
    k = cv2.waitKey(1) & 0xFF
    cv2.imshow("test",img)
    if k == 27:
           break