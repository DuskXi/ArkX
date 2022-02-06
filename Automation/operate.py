import re

from loguru import logger

from InterfaceLayer.interface import Interface


class Operate:
    def __init__(self, modelConfig, adb_path, labelsName):
        self.interface = Interface()
        self.objectModel = DictObject(**{"name": modelConfig["objectDetectionModel"]["modelName"],
                                         "path": modelConfig["objectDetectionModel"]["filePath"],
                                         "pbtxt": modelConfig["objectDetectionModel"]["pbtxt"]})
        self.sanityModel = DictObject(**{"name": modelConfig["addSanityModel"]["modelName"],
                                         "path": modelConfig["addSanityModel"]["filePath"],
                                         "pbtxt": modelConfig["addSanityModel"]["pbtxt"]})
        self.classifyModel = DictObject(**{"name": modelConfig["imageClassificationModel"]["modelName"],
                                           "path": modelConfig["imageClassificationModel"]["filePath"],
                                           "labels": modelConfig["imageClassificationModel"]["labels"]})
        self.ocrModel = DictObject(**{"det": modelConfig["ocrModel"]["det"],
                                      "cls": modelConfig["ocrModel"]["cls"],
                                      "rec": modelConfig["ocrModel"]["rec"]})
        self.adb_path = adb_path
        self.proxyName = modelConfig["proxyName"]
        self.labelsName = LabelsName(**labelsName)
        self.interface.setConfig(dynamicMemory=True)

    def initModel(self):
        self.interface.loadObjectDetectionModel(self.objectModel.name, self.objectModel.path, self.objectModel.pbtxt)
        self.interface.loadObjectDetectionModel(self.sanityModel.name, self.sanityModel.path, self.sanityModel.pbtxt)
        self.interface.loadImageClassificationModel(self.classifyModel.name, self.classifyModel.path, self.classifyModel.labels)
        self.interface.loadOcr(self.ocrModel.det, self.ocrModel.cls, self.ocrModel.rec)

    def loadGame(self):
        self.interface.loadAndroid(self.adb_path)

    def releaseGame(self):
        self.interface.releaseAndroid()

    def waitForGame(self):
        self.interface.waitScreen()

    def getGameStatus(self):
        result = self.interface.getClassification(self.classifyModel.name)
        return result[0]

    def _getObjectData(self, modelName):
        result = self.interface.detectObjectAndGetBox(modelName)
        boxes = []
        for index in range(len(result["boxes"])):
            boxes.append({
                "id": result["labels"][index]["id"],
                "name": result["labels"][index]["name"],
                "position": {
                    "xMin": result["boxes"][index][0],
                    "xMax": result["boxes"][index][1],
                    "yMin": result["boxes"][index][2],
                    "yMax": result["boxes"][index][3]
                },
                "percentage": result["percents"][index]
            })
        return boxes

    def getObjectDataMain(self):
        return self._getObjectData(self.objectModel.name)

    def getObjectDataSanity(self):
        return self._getObjectData(self.sanityModel.name)

    def getProxyStatus(self):
        result = self.interface.getProxyStatus(self.objectModel.name, self.proxyName)
        return result

    def getSanity(self):
        box = self._getBoxByName(self.objectModel.name, self.labelsName.IQ)
        if box is None:
            return "N/A"
        resolution = self.interface.getResolution()
        result = self.interface.runOcr([box["position"]["xMin"], resolution["x"], box["position"]["yMin"], box["position"]["yMax"]])
        if len(result) > 0:
            if "/" in result[0]["text"]:
                sanity, maxSanity = result[0]["text"].split("/")
                return int(sanity), int(maxSanity)
        return None

    def _getBoxByName(self, modelName, labelName):
        result = self._getObjectData(modelName)
        for box in result:
            if box["name"] == labelName:
                return box
        return None

    def getBoxMain(self, labelName):
        return self._getBoxByName(self.objectModel.name, labelName)

    def getBoxSanity(self, labelName):
        return self._getBoxByName(self.sanityModel.name, labelName)

    def getResolution(self):
        return self.interface.getResolution()

    def getDeviceName(self):
        return self.interface.getDeviceName()

    def getLevelSanity(self, startLabelName):
        result = self.getBoxMain(startLabelName)
        if result is not None:
            resolution = self.interface.getResolution()
            texts = self.interface.runOcr([result["position"]["xMin"], resolution["x"], result["position"]["yMin"], resolution["y"]])
            for text in texts:
                if re.match(r"[0-9\-]+", text["text"]):
                    number = int(text["text"])
                    number = number if number >= 0 else -1 * number
                    return number
        return None

    def getRunningInfoInWorking(self):
        results = self.runOcr()
        for text in results:
            if re.match(r"^[0-9]{1,3}/[0-9]{1,3}$", text["text"]):
                info = text["text"].split("/")
                return int(info[0]), int(info[1])
        return None

    def runOcr(self):
        resolution = self.interface.getResolution()
        result = self.interface.runOcr([0, resolution["x"], 0, resolution["y"]])
        return result

    def _click(self, x, y):
        logger.info("点击 %s, %s" % (x, y))
        self.interface.click(x, y)

    def click(self, x, y):
        self._click(x, y)

    def clickArea(self, xMin, xMax, yMin, yMax):
        self._click(float(xMin + xMax) / 2, float(yMin + yMax) / 2)

    def clickBox(self, boxPosition: dict):
        self.clickArea(boxPosition["xMin"], boxPosition["xMax"], boxPosition["yMin"], boxPosition["yMax"])


class DictObject:
    def __init__(self, **entries):
        self.name = None
        self.path = None
        self.pbtxt = None
        self.labels = None
        self.det = None
        self.cls = None
        self.rec = None
        self.__dict__.update(entries)


class LabelsName:
    def __init__(self, **entries):
        self.IQ = None
        self.__dict__.update(entries)
