import math
import threading
import time

import cv2
import numpy as np
from loguru import logger

from NeuralNetworks import networklayer
from NeuralNetworks.networklayer import ConvolutionalNeuralNetworkCore as CNNCore
from Ocr.ocr import PDOcr
from Device.device import Device


class Interface:
    # noinspection PyTypeChecker
    def __init__(self):
        self.cnnCore = CNNCore()
        self.device: Device = None
        self.ocr = None
        self.isUseResize = True
        self.identify = None

    # 设置
    @staticmethod
    def setConfig(**kwargs):
        if kwargs.get("GPULimit", None) is not None:
            networklayer.set_force_use_cpu()
        if kwargs.get("Memory", 0) > 0:
            networklayer.set_gpu_memory(memory=kwargs.get("Memory", 0))
        if kwargs.get("dynamicMemory", False):
            networklayer.set_dynamic_gpu_memory_usage()

    # 加载
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # 加载对象检测模型
    def loadObjectDetectionModel(self, name, path, pbtxt):
        self.cnnCore.loadODModel(name, path, pbtxt)
        logger.info("load object detection model[ " + name + ":" + path + "]")

    # 加载图像分类模型
    def loadImageClassificationModel(self, name, path, labels):
        self.cnnCore.loadICModel(name, path, labels)
        logger.info("load classify model[ " + name + ":" + path + "]")

    # 加载手机控制器
    def loadAndroid(self, adb_path: str, device_name=None):
        logger.info("loadAndroid adb_path: " + adb_path)
        self.device = Device(adb_path, device_name=device_name)
        threading.Thread(target=self.device.keepScreenRefresh, args=()).start()

    # 加载光学文字识别模型
    def loadOcr(self, det, cls, rec):
        self.ocr = PDOcr((det, cls, rec))
        logger.info("load ocr model")
        ''

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 释放所有控制器资源
    def releaseAndroid(self):
        logger.info("release Android")
        self.device.isRun = False
        time.sleep(1)
        self.device.android.release()
        self.device = None

    # 检测
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # 从缓存获取现在屏幕类型

    def getClassificationFromBuffer(self, name):
        return self.cnnCore.getClassificationFromBuffer(name)

    # 获取现在屏幕类型
    def getClassification(self, name):
        return self.cnnCore.getClassification(name, self.device.readMirror())

    # 进行分类并且返回类型
    def classifyImages(self, name):
        return self.cnnCore.classifyImages(name, self.device.readMirror())

    # 检测对象返回原始数据
    def detectObject(self, name):
        image = self.device.readMirror()
        if self.isUseResize:
            image = Tools.resizeImage(image)
        return self.cnnCore.detectObject(name, image)

    # 缩放至指定大小并检测对象返回原始数据
    def resizeAndDetectObject(self, name: str, width: int, height: int):
        image = self.device.readMirror()
        if self.isUseResize:
            image = Tools.resizeImage(image)
        image = np.array(cv2.resize(image, (width, height)))
        return self.cnnCore.detectObject(name, image)

    # 检测对象并且返回对象坐标

    def detectObjectAndGetBox(self, name):
        logger.info("detectObject in model: " + name)
        self.cnnCore.detectObject(name, self.device.readMirror())
        return self.cnnCore.getBoxData(name, self.device.latestScreen.shape[1], self.device.latestScreen.shape[0])

    # 从缓存获取对象坐标
    def getBoxCoordinate(self, name):
        """
        从缓存获取对象坐标
        """
        return self.cnnCore.getBoxData(name, self.device.latestScreen.shape[1], self.device.latestScreen.shape[0])

    # 获取绘制后的图像
    def getDrawBoxImage(self, name):
        image = self.device.readMirror()
        if self.isUseResize:
            image = Tools.resizeImage(image)
        image = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.cnnCore.drawBox(name, image)
        return image.copy()

    # 获取代理勾选状态
    # noinspection PyBroadException
    def getProxyStatus(self, name, proxyName):
        try:
            img = Tools.getSubImage(
                self.device.latestScreen, self.detectObjectAndGetBox(name), proxyName)
        except Exception:
            return True
        if img is None:
            return None
        img_bin = Tools.binarization(img)
        status = round(np.mean(img_bin) / 255)
        if status == 1:
            return True
        else:
            return False

    # 加载
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def getResolution(self):
        if self.device is None:
            return {"x": -1, "y": - 1}
        if type(self.device.latestScreen) != np.ndarray:
            return {"x": -1, "y": - 1}
        else:
            return {"x": self.device.latestScreen.shape[1], "y": self.device.latestScreen.shape[0]}

    def getDeviceName(self):
        return self.device.getDeviceName() if self.device is not None else "UNKNOWN"

    # 点击
    def click(self, x, y):
        self.device.click(x, y)

    def waitScreen(self):
        while self.device.latestScreen is None:
            time.sleep(0.01)

    def checkAndroidStatus(self):
        if self.device.android.is_video_socket_close:
            return False
        else:
            return True

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Ocr
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def runOcr(self, area: list):
        (xMin, xMax, yMin, yMax) = area
        result = self.ocr.detectByArea(np.array(self.device.latestScreen), xMax=xMax, xMin=xMin, yMax=yMax, yMin=yMin)
        return result


class Tools:

    # 缩放图片
    @staticmethod
    def resizeImage(image):
        (width, height) = Tools.getResizeInfo((image.shape[1], image.shape[0]))
        variationWidth = math.ceil(width / 2)
        variationHeight = math.ceil(height / 2)
        newImage = np.zeros((image.shape[0] + (variationHeight * 2), image.shape[1] + (variationWidth * 2), image.shape[2]), dtype=image.dtype)
        newImage[:, :, :] = 255
        newImage[variationHeight:variationHeight + image.shape[0], variationWidth:variationWidth + image.shape[1], :] = image
        return newImage

    # 获取缩放信息
    @staticmethod
    def getResizeInfo(resolution, resizeProportion=(16, 9)):
        (width, height) = resizeProportion
        (originalWidth, originalHeight) = resolution
        objectiveHeight = (originalWidth / width) * height
        if objectiveHeight >= originalHeight:
            return 0, objectiveHeight - originalHeight
        else:
            objectiveWidth = (originalHeight / height) * width
            return objectiveWidth - width, 0

    @staticmethod
    def getSubImage(image, ranges, name: str):
        for coordinates, label in zip(ranges["boxes"], ranges["labels"]):
            if label["name"] == name:
                (xMin, xMax, yMin, yMax) = coordinates
                (xMin, xMax, yMin, yMax) = (
                    int(xMin), int(xMax), int(yMin), int(yMax))
                image = np.array(image)
                subImage = image[yMin:yMax, xMin:xMax, :]
                return subImage

    @staticmethod
    def binarization(image):
        image = np.array(image)
        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return thresh
