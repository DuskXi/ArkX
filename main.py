import os

import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fileRead(fileName, encoding='utf-8'):
    with open(fileName, encoding=encoding) as f:
        return f.read()


def main():
    from Automation.distributor import Distributor
    from Performance import recoder
    from WebInterface import web

    modelConfig = json.loads(fileRead("config/model.json"))
    labelsName = json.loads(fileRead("config/labelsName.json"))
    config = json.loads(fileRead("config/config.json"))

    recoder.Recoder.debug = False
    recoder.Recoder.debugSleepingTime = 60 * 60
    recoder.Recoder.initDataSet([modelConfig["objectDetectionModel"]["modelName"], modelConfig["addSanityModel"]["modelName"]],
                                [modelConfig["imageClassificationModel"]["modelName"]])

    distributor = Distributor(modelConfig, config["adb_path"], labelsName)
    web.run(distributor, config)


if __name__ == "__main__":
    main()
