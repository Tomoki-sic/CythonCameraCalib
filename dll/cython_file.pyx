# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# distutils: include_dirs= ["C:\\Users\\Tomoki\\anaconda3\\envs\\cy37\\lib\\site-packages\\numpy\\core\\include"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

cimport numpy as cnp
import numpy as np
import math as math 

from libcpp.pair cimport pair

ctypedef long long LL

#四角形の中にある点が含まれているか判定する関数
#引数が(y,x)な点に注意
def inRect(point1, point2, point3, point4, target):
    vector_a = np.array([point1[1],point1[0]])
    vector_b = np.array([point3[1],point3[0]])
    vector_c = np.array([point4[1],point4[0]])
    vector_d = np.array([point2[1],point2[0]])
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

def bilinearInterpolation(self,p1,p2,p3,p4,p_ref,pixcel_value1,pixcel_value2,pixcel_value3,pixcel_value4):
    mean_x = abs(p4[0]+p2[0]-p1[0]-p3[0])
    mean_y = abs(p4[1]+p3[1]-p1[1]-p2[1])
    th = 0.95
    if mean_x == 0:
        mat1 = np.array([0.5,0.5])
    else:
        mat1_1 = abs(p4[0]+p2[0]-p_ref[0]*2)/mean_x
        mat1_2 = abs(p3[0]+p1[0]-p_ref[0]*2)/mean_x
        #周囲の４点がスキューの場合の対策
        if mat1_1>th or mat1_2>th:
            mat1_2 = 0
            mat1_1 = 0

        mat1 = np.array([mat1_1,mat1_2])
    if mean_y == 0:
        mat3 = np.array([0.5,0.5])
    else:
        mat1_1 = abs(p1[1]+p2[1]-p_ref[1]*2)/mean_y
        mat1_2 = abs(p3[1]+p4[1]-p_ref[1]*2)/mean_y
        #周囲の４点がスキューの場合の対策
        if mat1_1>th or mat1_2>th:
            mat1_2 = 0
            mat1_1 = 0

        mat3 = np.array([mat1_1,mat1_2])
    mat2_x = np.array([[pixcel_value3[0],pixcel_value1[0]],[pixcel_value4[0],pixcel_value2[0]]])
    mat2_y = np.array([[pixcel_value3[1],pixcel_value1[1]],[pixcel_value4[1],pixcel_value2[1]]])
    a1 = np.dot(mat1,mat2_x)
    dst_x = np.dot(a1,mat3)
    b1 = np.dot(mat1,mat2_y)
    dst_y = np.dot(b1,mat3)
    return dst_x, dst_y

def __nearestOnePoints(self, x_ref,y_ref,x_undist,y_undist,shape=(7,6)):
    idx = 0
    points = [0,0,0,0]
    width,height = shape
    minimum = math.hypot(x_undist[0]-x_ref, y_undist[0]-y_ref)
    for i in range(len(x_undist)):
        a = math.hypot(x_undist[i]-x_ref, y_undist[i]-y_ref)
        if a < minimum:
            minimum = a
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

def __nearestFourPointsFromBector(self, x_ref,y_ref,x_undist,y_undist,shape=(7,6)):
    point = __nearestOnePoints(x_ref,y_ref,x_undist,y_undist,shape=shape)
    p1,p2,p3,p4 = self.__getRect(point-shape[0]-1,x_undist,y_undist,shape)
    target = [x_ref,y_ref]
    print(target,p1,p2,p3,p4,self.myfunc.inRect(p1,p2,p3,p4,target))
    if self.myfunc.inRect(p1,p2,p3,p4,target) == True:
        print(p1,p2,p3,p4)
        return True, [p1,p2,p3,p4]
    p1,p2,p3,p4 = self.__getRect(point-shape[0]-1,x_undist,y_undist,shape)
    if self.myfunc.inRect(p1,p2,p3,p4,target) == True:
        print(p1,p2,p3,p4)
        return True, [p1,p2,p3,p4]
    p1,p2,p3,p4 = self.__getRect(point-1,x_undist,y_undist,shape)
    if self.myfunc.inRect(p1,p2,p3,p4,target) == True:
        print(p1,p2,p3,p4)
        return True, [p1,p2,p3,p4]
    p1,p2,p3,p4 = self.__getRect(point,x_undist,y_undist,shape)
    if self.myfunc.inRect(p1,p2,p3,p4,target) == True:
        print(p1,p2,p3,p4)
        return True, [p1,p2,p3,p4]
    return False, []

def  InterpolateLookUpTable(x_undist,y_undist,x_dist,y_dist,src,shape=(7,6)):
    h,w = src.shape[:2]
    coordinate = []
    block_x = np.ones((h,w),np.float32) * 0
    block_y = np.ones((h,w),np.float32) * 0
    x_range_min = max(x_undist[0:][::shape[0]])
    x_range_max = min(x_undist[shape[0]-1:][::shape[0]])
    y_range_min = max(y_undist[0:shape[0]])
    y_range_max = min(y_undist[shape[0]*(shape[1]-1)+1:])
    for y_ref in range(y_range_min,y_range_max):
        for x_ref in range(x_range_min,x_range_max):
            idx = __nearestFourPointsFromBector(x_ref,y_ref,x_undist,y_undist,shape=shape)
            p1 = (x_undist[idx[0]],y_undist[idx[0]])
            p2 = (x_undist[idx[1]],y_undist[idx[1]])
            p3 = (x_undist[idx[2]],y_undist[idx[2]])
            p4 = (x_undist[idx[3]],y_undist[idx[3]])
            p_ref = (x_ref,y_ref)
            pixcel_value1 = (x_dist[idx[0]],y_dist[idx[0]])
            pixcel_value2 = (x_dist[idx[1]],y_dist[idx[1]])
            pixcel_value3 = (x_dist[idx[2]],y_dist[idx[2]])
            pixcel_value4 = (x_dist[idx[3]],y_dist[idx[3]])
            dst_x, dst_y = bilinearInterpolation(p1,p2,p3,p4,p_ref,pixcel_value1,pixcel_value2,pixcel_value3,pixcel_value4)
            block_x[y_ref,x_ref] = dst_x
            block_y[y_ref,x_ref] = dst_y
    return block_x, block_y
