import pickle
import csv
import os
import pandas as pd
import numpy as np
import cv2

class setPram():
    def __init__(self,camera, K, dist):
        self.__camera = camera
        self.__K = self.loadPickle(K)
        self.__dist = self.loadPickle(dist)

    def getCameraName(self):
        return self.__camera

    def getMtx(self):
        return self.__K

    def getDist(self):
        return self.__dist

    def loadPickle(self, filename):
        with open(filename,'rb') as f:
            data = pickle.load(f)
            f.close()
            return data

    def writePickle(self,filename, data):
        with open(filename,'wb') as f:
            pickle.dump(data,f)
            f.close()

    def loadLookUpTable(self, file_name):
        dataSet = pd.read_csv(file_name)
        u_undist = dataSet["x_undist"].values.tolist()
        v_undist = dataSet["y_undist"].values.tolist()
        u_dist = dataSet["x_dist"].values.tolist()
        v_dist = dataSet["y_dist"].values.tolist()
        return u_undist, v_undist, u_dist, v_dist

    def saveLookUpTable(self,filename,points_ref,points_undist, points_dist):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x_ref","y_ref","x_undist", "y_undist", "x_dist","y_dist"])
            for ref, undist,dist in zip(points_ref,points_undist,points_dist):
                writer.writerow([ref[0],ref[1],undist[0], undist[1], dist[0], dist[1]])

    def saveComplementedImage(self, file_name, w, h, x_complemented, y_complemented):
        x_complemented = x_complemented/w*255
        y_complemented = y_complemented/h*255
        x_complemented = x_complemented.astype(np.uint8)
        y_complemented = y_complemented.astype(np.uint8)
        cv2.imwrite(file_name+"_x.bmp",x_complemented)
        cv2.imwrite(file_name+"_y.bmp",y_complemented)
