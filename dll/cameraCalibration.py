from turtle import xcor
import cython
import numpy as np
import cv2
import math
import glob

from . import imageProcessing

#カメラ行列、歪み行列から歪み補正後の画像を獲得する関数
def cameraCalibrateFromKandDist(src,mtx,dist, dst_size = (1552, 2080)):
    #入力画像の縦幅、横幅
    h_src, w_src = src.shape[:2]
    #出力画像のサイズ
    h_dst, w_dst = dst_size[0], dst_size[1]
    #空の画像の作成(出力用の画像)
    block = np.ones((h_dst,w_dst,3),np.uint8) * 0
    #カメラ行列と切り取り領域の獲得
    newcameramtx1, roi1=cv2.getOptimalNewCameraMatrix(mtx,dist,(w_src,h_src),1,(w_src, h_src)) 
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w_src,h_src),1,(w_src//2, h_src//2))
    #カメラ行列から光学中心の位置を獲得する
    cx, cy = newcameramtx[0,2], newcameramtx[1,2]
    dst_cx, dst_cy = newcameramtx1[0,2],newcameramtx1[1,2]
    x,y,w,h = roi
    undistort_img = cv2.undistort(src, mtx, dist, None, newcameramtx)
    block[int(dst_cy-abs(cy-y)):int(dst_cy+abs(y+h-cy)),int(dst_cx-int(abs(cx-x))):int(dst_cx+abs(x+w-cx))] = undistort_img[y:y+h, x:x+w]
    #block[y*2:y*2+h, x*2:x*2+w] = undistort_img[y:y+h, x:x+w]
    #block[dst_cy-int(h/2):dst_cy+int(h/2)+1, dst_cx-int(w/2):dst_cx+int(w/2)] = undistort_img[y:y+h, x:x+w]
    cv2.imwrite("test.jpg",block)
    return block

def cameraCalibrateFromLookUpTable(img,table_x, table_y,w,h):
    dst = imageProcessing.cameraCalibrateFromLookUpTable(img,table_x, table_y,w,h)
    return dst

def InterpolateLookUpTable(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape):
    return imageProcessing.InterpolateLookUpTable(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape)

def InterpolateLookUpTableBycubic(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape):
    return imageProcessing.InterpolateLookUpTableBycubic(u_undist, v_undist,u_dist,v_dist,DST_WIDTH, DST_HEIGHT, shape)

def calculateAffineTransformationMatrix(points1,points2):
    if len(points1) != len(points2):
        return False, []
    else:
        pt1_arr, pt2_arr = __createMatrixForAffine(points1,points2)
        A = np.linalg.pinv(pt1_arr)
        dst = np.matmul(A, pt2_arr).tolist()
        dst = [[dst[0][0],dst[1][0],dst[2][0]],[dst[3][0],dst[4][0],dst[5][0]],[0,0,1]]
        return True, np.array(dst)

def __createMatrixForAffine(points1,points2):
    dst1 = []
    dst2 = []
    for pt in points1:
        dst1.append([pt[0],pt[1],1,0,0,0])
        dst1.append([0,0,0,pt[0],pt[1],1])
    for pt in points2:
        dst2.append([pt[0]])
        dst2.append([pt[1]])
    return np.array(dst1),np.array(dst2)

def AffineTransformation(points,affine):
    dst=[]
    for pt in points:
        pt = np.array([[pt[0]],[pt[1]],[1]])
        dst.append(np.matmul(affine, pt).tolist())
    dst=np.uint16(np.around(dst))
    return dst

def createReferencePoint(x,y):
    arr = np.ones((x*y,2))
    for i in range(y):
        for j in range(x):
            arr[i*x+j,0] = j+1
            arr[i*x+j,1] = i+1
    return arr

def calculateUndistortedPoint(img,referencePoint,circleGrid,threshold):
    circles = labeling(img,threshold=threshold)
    circles = pickUpCircles(img,circles.tolist())
    ref_point = createReferencePoint(referencePoint[0],referencePoint[1])
    ret, affine = calculateAffineTransformationMatrix(ref_point,circles)
    print(affine)
    grid = createReferencePoint(circleGrid[0],circleGrid[1])
    est = AffineTransformation(grid,affine)
    dst = drawCircles(img,est)
    cv2.imwrite("test_affine.bmp",dst)
    est2 = []
    for point in est:
        est2.append([point[0][0],point[1][0]])
    return est2, grid

def createLookUpTable(u_undist, v_undist,u_dist,v_dist,shape=(7,6),len_undist=(2080,1552),len_dist=(2080,1552)):
    block_x = np.ones((len_undist[1],len_undist[0]),np.float32) * 0
    block_y = np.ones((len_undist[1],len_undist[0]),np.float32) * 0


#円検出を行う関数(img:検出する画像, dp:投影機の解像度, minDist:円の距離の閾値, edgeThrethold:エッジ検出の閾値, circleThrethold=円検出の閾値,minRadius:円の小ささの閾値,maxRadiud:円の大きさの閾値)
#戻り値(circle:[[中心点のx座標, 中心点のy座標, 円の半径],[中心点のx座標, 中心点のy座標, 円の半径],...])
def detectCircles(img,dp=1,minDist=20,edgeThrethold=100,circleThrethold=20,minRadius=0,maxRadius=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=edgeThrethold, param2=circleThrethold, minRadius=minRadius, maxRadius=maxRadius)
    if isinstance(circles, np.ndarray):
        circles = np.uint16(np.around(circles))
        return circles[0]
    else:
        return []

def labeling(img,threshold=100,rect_min=3,rect_max=200,value=255,show=False):
    points = []
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)[:,:,2]
    ret, img_thresh = cv2.threshold(gray, threshold, value, cv2.THRESH_BINARY_INV)
    if show:
        cv2.namedWindow("labeling", cv2.WINDOW_NORMAL)
        cv2.imshow("labeling",img_thresh)
    ret, labels, states, centers = cv2.connectedComponentsWithStats(img_thresh)
    if ret == False:
        return points
    for c,rect in zip(centers,states):
        if rect[2]>rect_min and rect[2]<rect_max:
            points.append([c[0],c[1]])
    points = np.uint16(np.around(points))
    return points


#円の描画を行う関数
def drawCircles(src,circles,F=1):
    img = src.copy()
    circle_len = 10
    if src.shape[1] < 2000:
        circle_len = 3 
    for circle in circles:
        if len(circle)>2:
            cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 165, 255), circle_len)
        cv2.circle(img, (circle[0], circle[1]), 0, (0, 255, 255),circle_len)
    return img

def pickUpDistortedPoints(img,threshold,F):
    circles = labeling(img,threshold=threshold)
    points_dist = pickUpCircles(img,circles.tolist(),F)
    return points_dist

#点をピックアップするための関数
def pickUpCircles(src,points,F=1):
    dst=[]
    f = 1/F
    a = len(points)
    copy = src.copy()
    input_img_name='pickUpCircles'
    cv2.namedWindow(input_img_name, cv2.WINDOW_NORMAL)
    copy = drawCircles(copy,points)

    mouse = mouseParam(input_img_name)
    while True:
        k = cv2.waitKey(1)
        if k == 27 or len(points)<=0:
            break
        elif k == ord('s'):
            copy = src.copy()
            x_ref, y_ref = mouse.getPos()
            idx = mouse.nearestPoint(points,x_ref,y_ref)
            dst.append([int(points[idx][0]*f),int(points[idx][1]*f)])
            points.pop(idx)
            copy = drawCircles(copy,points)
        elif k == ord('x'):
            copy = src.copy()
            x_ref, y_ref = mouse.getPos()
            points.append([int(x_ref), int(y_ref)])    
            copy = drawCircles(copy,points)
        elif k == ord("z"):
            copy = src.copy()
            points.append(dst.pop(-1))
            copy = drawCircles(copy,points)
        elif k == ord("a"):
            dst.append([0,0])
        #copy = cv2.putText(copy,str(len(dst)),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)()
        copy_h = cv2.cvtColor(copy,cv2.COLOR_RGB2HSV)[:,:,2]
        cv2.imshow(input_img_name,copy_h)
    copy = drawCircles(src.copy(),dst)
    cv2.imwrite("pick"+str(a)+".bmp",copy)
    
    return dst

    #点をピックアップするための関数(複数)
def pickUpCirclesAndSort(src,points,points2,file_name):
    dst1, dst2=[],[]
    copy = src.copy()
    input_img_name='pickUpCircles2'
    cv2.namedWindow(input_img_name, cv2.WINDOW_NORMAL)
    copy = drawCircles(copy,points)
    cv2.imshow(input_img_name,copy)
    cv2.imwrite(file_name+".bmp",copy)

    mouse = mouseParam(input_img_name)
    while True:
        k = cv2.waitKey(1)
        if k == ord('q') or len(points)<=0:
            break
        elif k == ord('s'):
            copy = src.copy()
            x_ref, y_ref = mouse.getPos()
            idx = mouse.nearestPoint(points,x_ref,y_ref)
            dst1.append([points[idx][0],points[idx][1]])
            dst2.append([points2[idx][0],points2[idx][1]])
            points.pop(idx)
            points2.pop(idx)
            copy = drawCircles(copy,points)
        cv2.imshow(input_img_name,copy)
    copy = src.copy()
    copy = drawCircles(src.copy(),dst1)
    cv2.imwrite("pick.bmp",copy)
    
    return dst1,dst2

def SynthesizeTable(table_x_dir, table_y_dir,img_width, img_height,set):
    table_x = np.ones((img_height,img_width),np.float32) * 0
    table_y = np.ones((img_height,img_width),np.float32) * 0
    count_x = np.ones((img_height,img_width),np.float32) * 0
    count_y = np.ones((img_height,img_width),np.float32) * 0
    x_file_name = glob.glob(table_x_dir+"*.binary")
    y_file_name = glob.glob(table_y_dir+"*.binary")
    for x_name, y_name in zip(x_file_name, y_file_name):
        #バイナリデータの読込
        complemented_x = set.loadPickle(x_name)
        complemented_y = set.loadPickle(y_name)
        #テーブルを足し合わせる
        table_x += complemented_x
        table_y += complemented_y
        #平均取るようの配列
        count_x[complemented_x>0] += 1
        count_y[complemented_y>0] += 1
    count_x[count_x==0] = 1
    count_y[count_y==0] = 1
    dst_x = table_x/count_x
    dst_y = table_y/count_y
    dst_x[count_x<2] = 0
    dst_y[count_y<2] = 0
    return dst_x, dst_y

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)

    def __CallBackFunc(self, eventType, x, y, flags, params):
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType
        self.mouseEvent["flags"] = flags

    def getData(self):
        return self.mouseEvent

    def getEvent(self):
        return self.mouseEvent["event"]

    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])

    def nearestPoint(self,points,x_ref,y_ref):
        minimum = math.hypot(points[0][0]-x_ref,points[0][1]-y_ref)
        idx = 0
        for i, point in enumerate(points):
            d = math.hypot(point[0]-x_ref,point[1]-y_ref)
            if d < minimum:
                minimum = d
                idx = i
        return idx
    


