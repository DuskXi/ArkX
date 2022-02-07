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

logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)


def fileRead(fileName, encoding="utf-8"):
    with open(fileName, encoding=encoding) as f:
        return f.read()


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
    frequency = int(data['frequency'])
    sanityTimes = int(data['sanityTimes'])
    useStone = bool(data['useStone'])
    taskType = data['type']
    if taskType == 'single':
        distributor.taskBuffer.taskType = 'single'
        distributor.taskBuffer.singleTask.frequency = frequency
        distributor.taskBuffer.singleTask.sanityTimes = sanityTimes
        distributor.taskBuffer.singleTask.useStone = useStone
        distributor.saveTaskConfig()
    else:
        intervalTime = int(data['intervalTime'])
        minStartMultiple = int(data['minStartMultiple'])
        distributor.taskBuffer.taskType = 'continuousTask'
        distributor.taskBuffer.continuousTask.intervalTime = intervalTime
        distributor.taskBuffer.continuousTask.inlineSingleTask.frequency = frequency
        distributor.taskBuffer.continuousTask.inlineSingleTask.sanityTimes = sanityTimes
        distributor.taskBuffer.continuousTask.inlineSingleTask.useStone = useStone
        distributor.taskBuffer.continuousTask.minStartMultiple = minStartMultiple
        distributor.saveTaskConfig()


@socketio.on('StopTask')
def stopTask():
    distributor.stopTask()


@socketio.on('LoadDevice')
def loadDevice():
    distributor.initDevice()


@socketio.on('UpdateScreenInfo')
def updateScreenInfo():
    distributor.updateScreenInfo()


@socketio.on('RequestInformationUpdate')
def requestInformationUpdate():
    updateInformation(distributor.getInformation())


def updateInformation(data):
    socketio.emit("InformationUpdate", data)


def notification(message):
    socketio.emit("Notification", message)


def eventTaskEnd():
    socketio.emit("TaskEnd")
    updateInformation(distributor.getInformation())


@app.route("/")
@app.route("/Index")
@app.route("/index")
def index():
    return fileRead("WebInterface/templates/index.html")


def run(_distributor: Distributor):
    global distributor
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
