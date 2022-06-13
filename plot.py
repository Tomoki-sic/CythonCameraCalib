# PYTHON_MATPLOTLIB_3D_PLOT_03

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import glob

from dll import setting
from dll import flyCapture
from dll import imageProcessing
from dll import cameraCalibration as cal

CAMERA = "STSD2_002"
K = "./"+CAMERA+"/cal9/K.pkl"
DIST = "./"+CAMERA+"/cal9/dist.pkl"
OUTPUT = "./"+CAMERA+"/circleGrid6/"
set = setting.setPram(CAMERA,K,DIST)

IMAGE_WIDTH = 2080
IMAGE_HEIGHT = 1552
DST_WIDTH = 2080
DST_HEIGHT = 1552
SHAPE = (10,9)
F = 0.25


x=[]
y=[]
u=[]
v=[]

"""
csv_files = glob.glob(OUTPUT+'*.csv')
for fname in csv_files:
    u_undist, v_undist, u_dist, v_dist = set.loadLookUpTable(fname)

    x += u_undist
    y += v_undist
    u += u_dist
    v += v_dist

#axオブジェクト作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#軸ラベル
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("v")

#データのプロット
#ax.scatter(x, y, u, "o", color="blue",s=1)
ax.scatter(x, y, v, "o", color="red",s=1)
plt.show()
"""






# Figureと3DAxeS
ax = Axes3D(plt.figure())
# 軸ラベルを設定
ax.set_xlabel("x", size = 16)
ax.set_ylabel("y", size = 16)
ax.set_zlabel("u", size = 16)

dst_x, dst_y = cal.SynthesizeTable(OUTPUT+"x/",OUTPUT+"y/",DST_WIDTH,DST_HEIGHT,set)
x = np.arange(dst_x.shape[1])
y = np.arange(dst_x.shape[0])

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = Axes3D(fig)
scat = ax.scatter(X, Y, dst_x)
plt.show()

"""

X = np.array(list(range(DST_WIDTH))*DST_HEIGHT)
Y = np.array(list(range(DST_HEIGHT))*DST_WIDTH)
z_x = dst_x.flatten()
z_y = dst_y.flatten()
Z = z_x


x_new, y_new = np.meshgrid(np.unique(X), np.unique(Y))
z_new = griddata((X, Y), Z, (x_new, y_new))
print(X.shape)
print(Y.shape)
print(Z.shape)

# 曲面を描画
ax.plot_surface(x_new, y_new, z_new)

# 底面に等高線を描画
ax.contour(x_new, y_new, z_new, colors = "black", offset = -1)

plt.show()

"""
