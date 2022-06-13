from . import PyCapture2
import numpy as np
import cv2

#pointGreyカメラで撮影を行うためのクラス----------------------------------------------------
class pointGreyCamera:
    def __init__(self,width,height):
        self.bus = PyCapture2.BusManager()
        self.cam = PyCapture2.Camera()
        self.uid =self.bus.getCameraFromIndex(0)
        self.cam.connect(self.uid)
        self.setImageSize(width,height)
        self.setCameraProperty()

    def setCameraProperty(self):
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHARPNESS, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, onOff = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, onOff = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.GAMMA, onOff = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHARPNESS, onOff = False)

    def setImageSize(self,width,height):
        self.fmt7info, supported = self.cam.getFormat7Info(0)
        offsetX = int((self.fmt7info.maxWidth-width)/2)
        offsetY = int((self.fmt7info.maxHeight-height)/2)
        self.pxfmt = PyCapture2.PIXEL_FORMAT.RAW8
        fmt7imgSet = PyCapture2.Format7ImageSettings(0, offsetX, offsetY, width, height, self.pxfmt)
        fmt7pktInf, isValid = self.cam.validateFormat7Settings(fmt7imgSet)
        self.cam.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)

    def startCapture(self):
        self.cam.startCapture()

    def stopCapture(self):
        self.cam.stopCapture()
        fmt7imgSet = PyCapture2.Format7ImageSettings(0, 0, 0, self.fmt7info.maxWidth, self.fmt7info.maxHeight, self.pxfmt)
        fmt7pktInf, isValid = self.cam.validateFormat7Settings(fmt7imgSet)
        self.cam.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHUTTER, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.GAIN, autoManualMode = True)
        self.cam.setProperty(type = PyCapture2.PROPERTY_TYPE.SHARPNESS, onOff = True)
        self.cam.disconnect()

#pointGrayCameraをopencvで扱うためのクラス
class pointGreyCamera2OpenCV(pointGreyCamera):
    def captureImage(self):
        try:
            tmp_image = self.cam.retrieveBuffer()
        except PyCapture2.Fc2error as fc2Err:
            print("Error retrieving buffer :", fc2Err)
            return False, []
        row_bytes = float(len(tmp_image.getData()))/float(tmp_image.getRows())
        cv_image = np.array(tmp_image.getData(), dtype="uint8").reshape((tmp_image.getRows(), tmp_image.getCols()));
        # 色空間を指定(これをしないとグレー画像になる)
        raw_image = cv2.cvtColor(cv_image, cv2.COLOR_BAYER_BG2BGR)
        return True, raw_image
