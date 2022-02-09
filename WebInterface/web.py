import json
import logging
import sys
import threading
import time
import webbrowser

from flask import Flask
from flask_socketio import SocketIO
from loguru import logger

from Automation.distributor import Distributor

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
# noinspection PyTypeChecker
distributor: Distributor = None
enableGPU = False
gpuMemoryLimit = 0
dynamicMemory = False
configDict = {}

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


def fileRead(fileName, encoding="utf-8"):
    with open(fileName, encoding=encoding) as f:
        return f.read()


def fileWrite(fileName, content, encoding="utf-8"):
    with open(fileName, "w", encoding=encoding) as f:
        f.write(content)


@socketio.on('StartTask')
def startTask(data):
    frequency = int(data['frequency'])
    sanityTimes = int(data['sanityTimes'])
    useStone = bool(data['useStone'])
    taskType = data['type']
    if taskType == 'single':
        distributor.newSingleTask(frequency, sanityTimes, useStone)
    else:
        intervalTime = int(data['intervalTime']) * 60
        minStartMultiple = int(data['minStartMultiple'])
        distributor.newContinuousTask(intervalTime, frequency, sanityTimes, useStone, minStartMultiple)


@socketio.on('ChangeTaskBuffer')
def changeTaskBuffer(data):
    taskType = data['type']
    try:
        if taskType == 'single':
            distributor.taskBuffer.taskType = 'single'
            distributor.taskBuffer.singleTask.frequency = int(data['singleTask']['frequency'])
            distributor.taskBuffer.singleTask.sanityTimes = int(data['singleTask']['sanityTimes'])
            distributor.taskBuffer.singleTask.useStone = bool(data['singleTask']['useStone'])
            distributor.saveTaskConfig()
        else:
            intervalTime = int(data['continuousTask']['intervalTime'])
            minStartMultiple = int(data['continuousTask']['minStartMultiple'])
            distributor.taskBuffer.taskType = 'continuousTask'
            distributor.taskBuffer.continuousTask.intervalTime = intervalTime
            distributor.taskBuffer.continuousTask.inlineSingleTask.frequency = int(data['continuousTask']['frequency'])
            distributor.taskBuffer.continuousTask.inlineSingleTask.sanityTimes = int(data['continuousTask']['sanityTimes'])
            distributor.taskBuffer.continuousTask.inlineSingleTask.useStone = bool(data['continuousTask']['useStone'])
            distributor.taskBuffer.continuousTask.minStartMultiple = minStartMultiple
            distributor.saveTaskConfig()
    except Exception as e:
        pass
        #logger.warning(f"JavaScript 提供的数据非法, 请忽略该条 {e} ")


@socketio.on('ChangeTaskBufferType')
def changeTaskBufferType(data):
    taskType = data['type']
    if taskType == 'single':
        distributor.taskBuffer.taskType = 'single'
    else:
        distributor.taskBuffer.taskType = 'continuousTask'


@socketio.on("ChangeGPUDeviceInfo")
def changeGPUDeviceInfo(data):
    global enableGPU
    global gpuMemoryLimit
    global dynamicMemory
    global configDict
    enableGPU = bool(data['enableGPU'])
    gpuMemoryLimit = float(data['gpuMemoryLimit'])
    dynamicMemory = bool(data['dynamicMemory'])
    configDict['EnableGPU'] = enableGPU
    configDict['GPUMemoryLimit'] = gpuMemoryLimit
    configDict['DynamicMemory'] = dynamicMemory
    fileWrite("config/config.json", json.dumps(configDict))


@socketio.on('StopTask')
def stopTask():
    distributor.stopTask()


@socketio.on('InitNeuralNetworks')
def initNeuralNetworks():
    global enableGPU
    global gpuMemoryLimit
    global dynamicMemory
    if enableGPU:
        distributor.initNeuralNetworks(enableGPU, gpuMemoryLimit if not dynamicMemory else "dynamic")
    else:
        distributor.initNeuralNetworks()


@socketio.on('LoadDevice')
def loadDevice():
    distributor.initDevice()


@socketio.on('ConnectToDevice')
def connectToDevice(data):
    device = data['device']
    distributor.initDevice(device)


@socketio.on('DisconnectDevice')
def disconnectDevice():
    distributor.disconnectDevice()


@socketio.on('UpdateScreenInfo')
def updateScreenInfo():
    distributor.updateScreenInfo()


@socketio.on('RequestInformationUpdate')
def requestInformationUpdate():
    updateInformation(distributor.getInformation())


@socketio.on('RequestSystemInformation')
def requestSystemInformation():
    socketio.emit("SystemInformation", systemInformation())


@socketio.on('RequestDevicesInformation')
def requestDevicesInformation():
    socketio.emit("DevicesInformation", devicesInformation())


def updateInformation(data):
    socketio.emit("InformationUpdate", data)


def notification(message):
    socketio.emit("Notification", message)


def eventTaskEnd():
    socketio.emit("TaskEnd")
    updateInformation(distributor.getInformation())


def systemInformation():
    global enableGPU
    global gpuMemoryLimit
    global dynamicMemory
    info = distributor.getSystemInfo()
    result = {**info, "enableGPU": enableGPU, "gpuMemoryLimit": gpuMemoryLimit, "dynamicMemory": dynamicMemory}
    return result


def devicesInformation():
    result = {}
    data = distributor.operate.getDevicesInfo()
    for device in data:
        result[device[0]] = device[1]
    return result


@app.route("/")
@app.route("/Index")
@app.route("/index")
def index():
    return fileRead("WebInterface/templates/index.html")


def run(_distributor: Distributor, config):
    global distributor
    global enableGPU
    global gpuMemoryLimit
    global dynamicMemory
    global configDict

    enableGPU = config['EnableGPU']
    gpuMemoryLimit = config['GPUMemoryLimit']
    dynamicMemory = config['DynamicMemory']
    configDict = config

    ws_handler = WSHandler()
    ws_handler.bindEvent(notification)
    logger.configure(handlers=[{"sink": ws_handler}, {"sink": sys.stdout}])
    logger.add("log/log_{time}.log", rotation="500MB", encoding="utf-8", enqueue=True, compression="zip", retention="10 days")
    distributor = _distributor
    distributor.bindTaskEndCallback(eventTaskEnd)

    threading.Thread(target=threadOpen).start()

    socketio.run(app)


def threadOpen():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")


class WSHandler:
    def __init__(self):
        self.sequence = []
        self.newData = None

    def write(self, data):
        self.sequence.append(data)
        if self.newData is not None:
            self.newData(data)

    def bindEvent(self, callback):
        self.newData = callback
