import datetime

import pytz


class Recoder:
    objectModel = {}
    classifyModel = {}
    ocrModel = []
    timezone = pytz.timezone('UTC')
    debug = False
    debugSleepingTime = 60 * 60

    @staticmethod
    def initDataSet(objectModelNames, classifyModelNames):
        for name in objectModelNames:
            Recoder.objectModel[name] = []
        for name in classifyModelNames:
            Recoder.classifyModel[name] = []
        Recoder.ocrModel = []
        Recoder.timezone = pytz.timezone('UTC')

    @staticmethod
    def printInfo():
        pass

    @staticmethod
    def recordObjectModel(modelName, length, percentage):
        for key in percentage:
            percentage[key] = float(percentage[key])
        Recoder.objectModel[modelName].append({
            "timeLong": length,
            "percentage": percentage,
            "datetime": datetime.datetime.now(Recoder.timezone).isoformat()
        })

    @staticmethod
    def recordClassifyModel(modelName, length, percentage):
        Recoder.classifyModel[modelName].append({
            "timeLong": length,
            "percentage": float(percentage),
            "datetime": datetime.datetime.now(Recoder.timezone).isoformat()
        })

    @staticmethod
    def recordOcr(length, percentage):
        Recoder.ocrModel.append({
            "timeLong": length,
            "percentage": percentage,
            "datetime": datetime.datetime.now(Recoder.timezone).isoformat()
        })

    @staticmethod
    def getDataset():
        return {
            "objectModel": Recoder.objectModel,
            "classifyModel": Recoder.classifyModel,
            "ocrModel": Recoder.ocrModel,
            "listObjectModel": list(Recoder.objectModel.keys()),
            "listClassifyModel": list(Recoder.classifyModel.keys())
        }
