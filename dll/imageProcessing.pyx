# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# distutils: include_dirs= ["C:\\Users\\Tomoki\\anaconda3\\envs\\cy36\\lib\\site-packages\\numpy\\core\\include"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
# load_ext cythonmagic

import numpy as np
cimport numpy as cnp
import math
import cv2
from libcpp.vector cimport vector

""""
cpdef cnp.ndarray[cnp.int8_t, ndim=3] cameraCalibrateFromKandDist(cnp.uint8_t[:,:,:] src, cnp.float64_t[:,:] K, cnp.float64_t[:,:] Dist, int dst_width, int dst_height):
    cdef:
        cnp.ndarray[cnp.uint8_t, ndim=3] img = np.asarray(src)
        cnp.ndarray[cnp.uint8_t, ndim=3] block = np.ones((dst_height,dst_width,3),np.uint8) * 0
        cnp.ndarray[cnp.uint8_t, ndim=3] undistort_img
        cnp.ndarray[cnp.float_t, ndim=2] mtx = np.asarray(K)
        cnp.ndarray[cnp.float_t, ndim=2] dist = np.asarray(Dist)
        int img_h = img.shape[0], img_w = img.shape[1]
        int dst_cx = <int>dst_width/2, dst_cy = <int>dst_height/2
    #カメラ行列と切り取り領域の獲得
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(img_w,img_h),1,(img_w,img_h))
    x,y,w,h = roi
    undistort_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    block[dst_cy-int(h/2):dst_cy+int(h/2)+1, dst_cx-int(w/2):dst_cx+int(w/2)] = undistort_img[y:y+h, x:x+w]
    return block

cpdef detectCirclesFromGGreenIMG(cnp.uint8_t[:,:,:] src, int dp=1, int minDist=20, int edgeThrethold=100, int circleThrethold=20, int minRadius=0, int maxRadius=0):
    cdef:
        cnp.ndarray[cnp.uint8_t, ndim=3] img = np.asarray(src)
        cnp.ndarray[cnp.uint8_t, ndim=2] gray = img[:,:,1]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=edgeThrethold, param2=circleThrethold, minRadius=minRadius, maxRadius=maxRadius)
    if isinstance(circles, np.ndarray):
        circles = np.uint16(np.around(circles))
        return circles[0]
    else:
        return []

#円の描画を行う関数
cpdef cnp.ndarray[cnp.int8_t, ndim=3] drawCircles(cnp.uint8_t[:,:,:] src, circles):
    cdef:
        cnp.ndarray[cnp.uint8_t, ndim=3] img
    img = np.asarray(src).copy()
    for circle in circles:
        img = cv2.circle(img, (circle[0], circle[1]), 0, (0, 0, 255),10)
    return img

"""
cpdef cnp.ndarray[cnp.uint8_t, ndim=3] cameraCalibrateFromLookUpTable(cnp.uint8_t[:,:,:] src, float[:,:] table_x, float[:,:] table_y, int w, int h):
    cdef:
        int y, x
        cnp.uint8_t[:,:,:] dst = np.arange(h*w*3, dtype="uint8").reshape((h, w, 3))
    for y in range(h):
        for x in range(w):
            if <int>table_y[y,x] == 0:
                dst[y,x,:] =  0   
            else:
                dst[y,x] = src[<int>table_y[y,x],<int>table_x[y,x]]
            
    return np.asarray(dst)

cpdef InterpolateLookUpTable(vector[int] x_undist, vector[int] y_undist, vector[int] x_dist, vector[int] y_dist,int img_w,int img_h,tuple shape=(7,6)):
    cdef:
        cnp.float32_t[:,:] block_x = np.arange(img_h*img_w, dtype="float32").reshape((img_h, img_w)) * 0
        cnp.float32_t[:,:] block_y = np.arange(img_h*img_w, dtype="float32").reshape((img_h, img_w)) * 0
        int x_range_min = max(x_undist[0:][::shape[0]])
        int x_range_max = min(x_undist[shape[0]-1:][::shape[0]])
        int y_range_min = max(y_undist[0:shape[0]])
        int y_range_max = min(y_undist[shape[0]*(shape[1]-1)+1:])
        int y_ref, x_ref
        vector[int] p1, p2, p3, p4, p_ref
        vector[int] pixcel_value1, pixcel_value2, pixcel_value3, pixcel_value4
    for y_ref in range(y_range_min,y_range_max):
        if y_ref%10==0:
            print(y_ref-y_range_max)
        for x_ref in range(x_range_min,x_range_max):
            ret, idx =  __nearestFourPointsFromBector(x_ref, y_ref, x_undist, y_undist,shape=shape)
            if ret:
                p1 = [x_undist[idx[0]],y_undist[idx[0]]]
                p2 = [x_undist[idx[1]],y_undist[idx[1]]]
                p3 = [x_undist[idx[2]],y_undist[idx[2]]]
                p4 = [x_undist[idx[3]],y_undist[idx[3]]]
                p_ref = [x_ref,y_ref]
                pixcel_value1 = [x_dist[idx[0]],y_dist[idx[0]]]
                pixcel_value2 = [x_dist[idx[1]],y_dist[idx[1]]]
                pixcel_value3 = [x_dist[idx[2]],y_dist[idx[2]]]
                pixcel_value4 = [x_dist[idx[3]],y_dist[idx[3]]]
                dst_x, dst_y = bilinearInterpolation(p1,p2,p3,p4,p_ref,pixcel_value1,pixcel_value2,pixcel_value3,pixcel_value4)
                block_x[y_ref,x_ref] = dst_x
                block_y[y_ref,x_ref] = dst_y
    return np.asarray(block_x), np.asarray(block_y)

cdef int __nearestOnePoints(int x_ref,int y_ref,vector[int] x_undist, vector[int] y_undist,tuple shape):
    cdef:
        int idx = 0
        int width=shape[0], height = shape[1]
        double minimum, minimum_ass
        int i,size = len(x_undist),idx_pre
    minimum = math.hypot(x_undist[0]-x_ref, y_undist[0]-y_ref)
    for i in range(size):
        minimum_ass =math.hypot(x_undist[i]-x_ref, y_undist[i]-y_ref)
        if minimum_ass < minimum:
            minimum = minimum_ass
            idx = i
    idx_pre = idx
    if idx_pre < width:
        idx += width
    if (idx_pre+1)%width == 1:
        idx += 1
    if (idx_pre+1)%width == 0:
        idx -= 1
    if idx_pre >= width*(height-1):
        idx -= width
    return idx

cpdef __nearestFourPointsFromBector(int x_ref,int y_ref,vector[int] x_undist,vector[int] y_undist,tuple shape=(7,6)):
    cdef:
        int point
        vector[int] p1, p2, p3, p4
        int idx1, idx2, idx3,idx4
    point = __nearestOnePoints(x_ref,y_ref,x_undist,y_undist,shape=shape)
    p1,p2,p3,p4,idx1, idx2, idx3,idx4 = __getRect(point-shape[0]-1,x_undist,y_undist,shape)
    target = [x_ref,y_ref]
    if inRect(p1,p2,p3,p4,target) == True:
        return True, [idx1, idx2, idx3,idx4]
    p1,p2,p3,p4,idx1, idx2, idx3,idx4 = __getRect(point-shape[0],x_undist,y_undist,shape)
    if inRect(p1,p2,p3,p4,target) == True:
        return True, [idx1, idx2, idx3,idx4]
    p1,p2,p3,p4,idx1, idx2, idx3,idx4 = __getRect(point-1,x_undist,y_undist,shape)
    if inRect(p1,p2,p3,p4,target) == True:
        return True, [idx1, idx2, idx3,idx4]
    p1,p2,p3,p4, idx1, idx2, idx3,idx4 = __getRect(point,x_undist,y_undist,shape)
    if inRect(p1,p2,p3,p4,target) == True:
        return True, [idx1, idx2, idx3,idx4]
    return False, []#四角形の中にある点が含まれているか判定する関数

#引数が(y,x)な点に注意
cpdef bint inRect_pre(vector[int] point1, vector[int] point2, vector[int] point3, vector[int] point4, vector[int] target):
    vector_a = np.array([point3[1],point3[0]])
    vector_b = np.array([point1[1],point1[0]])
    vector_c = np.array([point2[1],point2[0]])
    vector_d = np.array([point4[1],point4[0]])
    vector_e = np.array([target[1],target[0]])
    vector_ab = vector_b - vector_a
    vector_ae = vector_e - vector_a
    vector_bc = vector_c - vector_b
    vector_be = vector_e - vector_b
    vector_cd = vector_d - vector_c
    vector_ce = vector_e - vector_c
    vector_da = vector_a - vector_d
    vector_de = vector_e - vector_d
    vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
    vector_cross_bc_be = np.cross(vector_bc, vector_be)
    vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
    vector_cross_da_de = np.cross(vector_da, vector_de)
    return vector_cross_ab_ae <= 0 and vector_cross_bc_be <= 0 and vector_cross_cd_ce <= 0 and vector_cross_da_de <= 0

cpdef bint inRect(vector[int] point1, vector[int] point2, vector[int] point3, vector[int] point4, vector[int] target):
    cdef:
        int vector_ax = point3[1], vector_ay = point3[0]
        int vector_bx = point1[1], vector_by = point1[0]
        int vector_cx = point2[1], vector_cy = point2[0]
        int vector_dx = point4[1], vector_dy = point4[0]
        int vector_ex = target[1], vector_ey = target[0]
        int vector_abx = vector_bx - vector_ax, vector_aby = vector_by - vector_ay
        int vector_aex = vector_ex - vector_ax, vector_aey = vector_ey - vector_ay
        int vector_bcx = vector_cx - vector_bx, vector_bcy = vector_cy - vector_by
        int vector_bex = vector_ex - vector_bx, vector_bey = vector_ey - vector_by
        int vector_cdx = vector_dx - vector_cx, vector_cdy = vector_dy - vector_cy
        int vector_cex = vector_ex - vector_cx, vector_cey = vector_ey - vector_cy
        int vector_dax = vector_ax - vector_dx, vector_day = vector_ay - vector_dy
        int vector_dex = vector_ex - vector_dx, vector_dey = vector_ey - vector_dy
        int vector_cross_ab_ae = vector_abx*vector_aey - vector_aby*vector_aex 
        int vector_cross_bc_be = vector_bcx*vector_bey - vector_bcy*vector_bex
        int vector_cross_cd_ce = vector_cdx*vector_cey - vector_cdy*vector_cex
        int vector_cross_da_de = vector_dax*vector_dey - vector_day*vector_dex
    return vector_cross_ab_ae <= 0 and vector_cross_bc_be <= 0 and vector_cross_cd_ce <= 0 and vector_cross_da_de <= 0


cpdef bilinearInterpolation(vector[int] p1, vector[int] p2, vector[int] p3, vector[int] p4, vector[int] p_ref, vector[int] pixcel_value1, vector[int] pixcel_value2, vector[int] pixcel_value3, vector[int] pixcel_value4):
    mean_x = abs(p4[0]+p2[0]-p1[0]-p3[0])
    mean_y = abs(p4[1]+p3[1]-p1[1]-p2[1])
    if mean_x == 0:
        mat1 = np.array([0.5,0.5])
    else:
        mat1_1 = abs(p4[0]+p2[0]-p_ref[0]*2)/mean_x
        mat1_2 = abs(p3[0]+p1[0]-p_ref[0]*2)/mean_x
        mat1 = np.array([mat1_1,mat1_2])
    if mean_y == 0:
        mat3 = np.array([0.5,0.5])
    else:
        mat1_1 = abs(p1[1]+p2[1]-p_ref[1]*2)/mean_y
        mat1_2 = abs(p3[1]+p4[1]-p_ref[1]*2)/mean_y
        mat3 = np.array([mat1_1,mat1_2])
    mat2_x = np.array([[pixcel_value3[0],pixcel_value1[0]],[pixcel_value4[0],pixcel_value2[0]]])
    mat2_y = np.array([[pixcel_value3[1],pixcel_value1[1]],[pixcel_value4[1],pixcel_value2[1]]])
    a1 = np.dot(mat1,mat2_x)
    dst_x = np.dot(a1,mat3)
    b1 = np.dot(mat1,mat2_y)
    dst_y = np.dot(b1,mat3)
    return dst_x, dst_y

cpdef bilinearInterpolation2(vector[int] p1, vector[int] p2, vector[int] p3, vector[int] p4, vector[int] p_ref, vector[int] pixcel_value1, vector[int] pixcel_value2, vector[int] pixcel_value3, vector[int] pixcel_value4):
    cdef:
        float mean_x=abs(p4[0]+p2[0]-p1[0]-p3[0]), mean_y=abs(p4[1]+p3[1]-p1[1]-p2[1])
        float x_per1, x_per2, y_per1, y_per2
        float x_value1=pixcel_value1[0], x_value2=pixcel_value2[0], x_value3=pixcel_value3[0], x_value4=pixcel_value4[0]
        float y_value1=pixcel_value1[1], y_value2=pixcel_value2[1], y_value3=pixcel_value3[1], y_value4=pixcel_value4[1]
        float I_x, I_y
    if mean_x == 0:
        x_per1 = 0.5
        x_per2 = 0.5
    else:
        x_per1 = abs(p4[0]+p2[0]-p_ref[0]*2)/mean_x
        x_per2 = abs(p3[0]+p1[0]-p_ref[0]*2)/mean_x
        if x_per1 + x_per2 > 1:
            return 0,0
    if mean_y == 0:
        y_per1 = 0.5
        y_per2 = 0.5
    else:
        y_per1 = abs(p1[1]+p2[1]-p_ref[1]*2)/mean_y
        y_per2 = abs(p3[1]+p4[1]-p_ref[1]*2)/mean_y
        if y_per1 + y_per2 > 1:
            return 0,0
    I_x = x_per1*y_per1*x_value1 + x_per2*y_per1*x_value2 + x_per1*y_per2*x_value3 + x_per2*y_per2*x_value4
    I_y = x_per1*y_per1*y_value1 + x_per2*y_per1*y_value2 + x_per1*y_per2*y_value3 + x_per2*y_per2*y_value4 
    return I_x, I_y


cpdef InterpolateLookUpTableBycubic(vector[int] x_undist, vector[int] y_undist, vector[int] x_dist, vector[int] y_dist,int img_w,int img_h,tuple shape=(7,6)):
    cdef:
        cnp.float32_t[:,:] block_x = np.arange(img_h*img_w, dtype="float32").reshape((img_h, img_w)) * 0
        cnp.float32_t[:,:] block_y = np.arange(img_h*img_w, dtype="float32").reshape((img_h, img_w)) * 0
        int x_range_min = max(x_undist[0:][::shape[0]])
        int x_range_max = min(x_undist[shape[0]-1:][::shape[0]])
        int y_range_min = max(y_undist[0:shape[0]])
        int y_range_max = min(y_undist[shape[0]*(shape[1]-1)+1:])
        int y_ref, x_ref
        vector[int] p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p_ref
        vector[int] pixcel_value1, pixcel_value2, pixcel_value3, pixcel_value4
    for y_ref in range(y_range_min,y_range_max):
        if y_ref%10==0:
            print(y_ref-y_range_max)
        for x_ref in range(x_range_min,x_range_max):
            ret, idx =  __nearest16PointsFromBector(x_ref, y_ref, x_undist, y_undist,shape=shape)
            if ret:
                p_ref = [x_ref,y_ref]
                dst_x =  bicubicInterpolation(idx, p_ref, x_undist, y_undist, x_dist)
                dst_y =  bicubicInterpolation(idx, p_ref, x_undist, y_undist, y_dist)
                block_x[y_ref,x_ref] = dst_x
                block_y[y_ref,x_ref] = dst_y
    return np.asarray(block_x), np.asarray(block_y)

cpdef __nearest16PointsFromBector(int x_ref,int y_ref,vector[int] x_undist,vector[int] y_undist,tuple shape=(7,6)):
    ret, idx =  __nearestFourPointsFromBector(x_ref, y_ref, x_undist, y_undist,shape=shape)
    cdef bint p1_bool = idx[0] >= shape[0] and (idx[0]+1)%shape[0] != 1
    cdef bint p2_bool = (idx[1]+1)%shape[0] != 0 and idx[1] >= shape[0]
    cdef bint p3_bool = (idx[2]+1)%shape[0] != 1 and idx[2] < shape[0]*(shape[1]-1)
    cdef bint p4_bool = (idx[3]+1)%shape[0] != 0 and idx[3] < shape[0]*(shape[1]-1)
    cdef int p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16
    if ret and p1_bool and p2_bool and p3_bool and p4_bool:
        p1 = idx[0] - shape[0] - 1
        p2 = idx[0] - shape[0]
        p3 = idx[1] - shape[0]
        p4 = idx[1] - shape[0] + 1
        p5 = idx[0] - 1
        p6 = idx[0]
        p7 = idx[1]
        p8 = idx[1] + 1
        p9 = idx[2] - 1
        p10 = idx[2]
        p11 = idx[3]
        p12 = idx[3] + 1
        p13 = idx[2] + shape[0] - 1
        p14 = idx[2] + shape[0]
        p15 = idx[3] + shape[0]
        p16 = idx[3] + shape[0] + 1
        return True, [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]
    else:
        return False, []

cpdef float bicubicInterpolation(vector[int] idx, vector[int] p_ref, vector[int] x_undist, vector[int] y_undist, vector[int] dist):
    cdef float a = -1
    mean_x = x_undist[idx[6]] - x_undist[idx[5]]
    mean_y = y_undist[idx[9]] - y_undist[idx[5]]
    if mean_x == 0 or mean_y == 0:
        return 0
    
    x1 = (p_ref[0]-x_undist[idx[5]])/mean_x + 1
    x2 = (p_ref[0]-x_undist[idx[5]])/mean_x 
    x3 = (x_undist[idx[6]]-p_ref[0])/mean_x
    x4 = (x_undist[idx[6]]-p_ref[0])/mean_x + 1
    y1 = (p_ref[1]-y_undist[idx[5]])/mean_y + 1
    y2 = (p_ref[1]-y_undist[idx[5]])/mean_y 
    y3 = (y_undist[idx[9]]-p_ref[1])/mean_y
    y4 = (y_undist[idx[9]]-p_ref[1])/mean_y + 1 
    mat1 = np.array([__h(x1,a),__h(x2,a),__h(x3,a),__h(x4,a)])
    mat2 = np.array([[dist[idx[0]],dist[idx[4]],dist[idx[8]],dist[idx[12]]],\
            [dist[idx[1]],dist[idx[5]],dist[idx[9]],dist[idx[13]]],\
            [dist[idx[2]],dist[idx[6]],dist[idx[10]],dist[idx[14]]],\
            [dist[idx[3]],dist[idx[7]],dist[idx[11]],dist[idx[15]]]])
    mat3 = np.array([__h(y1,a), __h(y2,a),__h(y3,a),__h(y4,a)])
    return np.dot(np.dot(mat1, mat2),mat3)

    
cpdef float __h(float t, float a):
    abs_t = abs(t)
    if abs_t<=1:
        return (a+2)*abs_t*abs_t*abs_t-(a+3)*abs_t*abs_t + 1
    elif 1 < abs_t and abs_t <= 2:
        return a*abs_t*abs_t*abs_t - 5*a*abs_t*abs_t+8*a*abs_t-4*a
    else:
        return 0


cpdef __getRect(int point,vector[int] x_undist, vector[int] y_undist,tuple shape):
    cdef vector[int] point1, point2, point3, point4
    point1 = [x_undist[point],y_undist[point]]
    point2 = [x_undist[point+1],y_undist[point+1]]
    point3 = [x_undist[point+shape[0]],y_undist[point+shape[0]]]
    point4 = [x_undist[point+shape[0]+1],y_undist[point+shape[0]+1]]
    return point1, point2, point3, point4,point,point+1, point+shape[0], point+shape[0]+1


