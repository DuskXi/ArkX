import json
import os
import re
import hashlib
import threading
from File.file import File

from Performance import recoder
from .operate import Operate
from .task import *
import GPUtil
import tensorflow as tf
import psutil


# noinspection PyTypeChecker
class Distributor:
    def __init__(self, modelConfig, adb_path, labelsName, taskBufferPath="./config/taskBuffer.json"):
        self.operate = Operate(modelConfig, adb_path, labelsName)
        self.modelConfig = modelConfig
        self.taskBufferPath = taskBufferPath
        self.taskBuffer = TaskBuffer()
        self.taskBuffer.load(self.taskBufferPath)
        self.adb_path = adb_path
        self.labelsName = labelsName
        self.automation = None
        self.continuousTask = None
        self.mainThread: threading.Thread = None
        self.taskEndCallback = None
        self.taskType = "UNKNOWN"
        self.taskRecord = None
        # 初始化状态
        self.initIng = False
        self.neuralNetworksInited = False
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            self.operate.interface.setConfig(dynamicMemory=True)

    def initModelFile(self):
        icModelConfig = self.modelConfig["imageClassificationModel"]
        icPath = icModelConfig["dir"]
        icModelFileName = icModelConfig["fileName"]
        icModelSplitFiles = icModelConfig["files"]
        if not os.path.exists(os.path.join(icPath, icModelFileName)):
            file = File()
            file.mergedFile(icPath, icModelFileName, icModelSplitFiles)
            hash_str_cal = self.file_hash(os.path.join(icPath, icModelFileName), hashlib.sha256)
            with open(os.path.join(icPath, icModelFileName + ".sha256"), "r") as f:
                hash_str_source = f.read()
            if hash_str_cal != hash_str_source:
                raise Exception("文件合并失败")

    @staticmethod
    def file_hash(file_path: str, hash_method) -> str:
        if not os.path.isfile(file_path):
            print('文件不存在。')
            return ''
        h = hash_method()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def initDevice(self, device_name=None):
        self.initIng = True
        automation = Automation(self.operate, self.labelsName, None)
        automation.loadGame(device_name=device_name)
        self.automation = automation
        self.initIng = False

    def connectToIPAddress(self, ipAddress):
        self.operate.deviceManager.connectToIPAddress(ipAddress)

    def initNeuralNetworks(self, enableGPU=False, gpuMemoryLimit=None):
        self.initModelFile()
        if enableGPU:
            logger.info("已启用GPU加速模式")
            logger.info("正在检查GPU设备")
            sysInfo = self.getSystemInfo()
            if len(sysInfo["GPU"]) < 1:
                logger.error("未检测到GPU设备")
                logger.error("请检是否安装了对应的CUDA和cuDNN驱动")
                logger.error("无法启动GPU加速")
                logger.warning("切换为CPU模式")
                self.operate.interface.setConfig(GPULimit=True)
                self.operate.initModel()
            else:
                if gpuMemoryLimit == "dynamic":
                    self.operate.interface.setConfig(dynamicMemory=True)
                    logger.info("GPU设置为动态显存")
                else:
                    gpuMemoryLimit *= sysInfo["GPU"][0]["memoryValue"]
                    gpuMemoryLimit = int(gpuMemoryLimit)
                    self.operate.interface.setConfig(Memory=gpuMemoryLimit)
                    logger.info("GPU显存被限制为{}MB".format(gpuMemoryLimit))
                logger.debug(f"GPU设备: {sysInfo['GPU'][0]['description']} 已启用")
                self.operate.initModel(enableGPU=True, ocrMemory=2048)
        else:
            logger.info("初始化CPU设备...")
            self.operate.interface.setConfig(GPULimit=True)
            self.operate.initModel()

        self.neuralNetworksInited = True

    def newSingleTask(self, frequency, sanityTimes: int = 0, useStone: bool = False):
        task = Task(frequency, sanityTimes, useStone)
        self.taskRecord = task.copy()
        self.automation.reset(task)
        self.mainThread = threading.Thread(target=self.automation.mainLoop, args=(self.eventTaskEnd,))
        self.mainThread.start()
        self.taskType = "Single"

        self.taskBuffer.taskType = "single"
        self.taskBuffer.singleTask.frequency = frequency
        self.taskBuffer.singleTask.sanityTimes = sanityTimes
        self.taskBuffer.singleTask.useStone = useStone
        self.taskBuffer.save(self.taskBufferPath)

    def newContinuousTask(self, intervalTime, frequency, sanityTimes: int = 0, useStone: bool = False, minStartMultiple: int = 2):
        task = Task(frequency, sanityTimes, useStone)
        self.taskRecord = task.copy()
        self.automation.reset(task)
        self.continuousTask = ContinuousTask(self.automation, intervalTime, minStartMultiple)
        self.mainThread = threading.Thread(target=self.continuousTask.run, args=(self.eventTaskEnd,))
        self.mainThread.start()
        self.taskType = "Continuous"

        self.taskBuffer.taskType = "continuousTask"
        self.taskBuffer.continuousTask.intervalTime = int(intervalTime / 60)
        self.taskBuffer.continuousTask.inlineSingleTask.frequency = frequency
        self.taskBuffer.continuousTask.inlineSingleTask.sanityTimes = sanityTimes
        self.taskBuffer.continuousTask.inlineSingleTask.useStone = useStone
        self.taskBuffer.continuousTask.minStartMultiple = minStartMultiple
        self.taskBuffer.save(self.taskBufferPath)

    def stopTask(self):
        self.taskRecord = None
        if self.taskType == "Single":
            self.automation.stop()
        elif self.taskType == "Continuous":
            self.continuousTask.stop()

    def bindTaskEndCallback(self, callback):
        self.taskEndCallback = callback

    def eventTaskEnd(self):
        self.taskRecord = None
        self.taskEndCallback()
        self.taskType = "UNKNOWN"

    def updateScreenInfo(self):
        try:
            self.automation.updateScreenInfo()
        except Exception as e:
            logger.error(e)

    def saveTaskConfig(self):
        self.taskBuffer.save(self.taskBufferPath)

    def disconnectDevice(self):
        self.automation.operate.releaseDevices()

    def getInformation(self):
        return {
            "Performance": recoder.Recoder.getDataset(),
            "Task": (self.automation.task.getTaskInfo() if self.automation.task is not None else None) if self.automation is not None else None,
            "Screen": self.automation.screen if self.automation is not None else None,
            "Resolution": self.automation.operate.getResolution() if self.automation is not None else None,
            "DeviceName": self.automation.operate.getDeviceName() if self.automation is not None else None,
            "NeuralNetworksStatus": self.neuralNetworksInited,
            "ContinuousTask": self.continuousTask.status if self.continuousTask is not None else None,
            "TaskType": self.taskType,
            "TaskStatus": self.taskType if self.taskType == "UNKNOWN" else (str(self.automation.isRun) if self.taskType == "Single" else self.continuousTask.status),
            "LevelInfo": self.automation.getScreenInfo() if self.automation is not None else None,
            "FightProgress": self.automation.progress if self.automation is not None else None,
            "TaskProgress": [self.automation.task.frequency, self.taskRecord.frequency] if self.taskRecord is not None and self.automation is not None else None,
            "TaskBuffer": self.taskBuffer.getAsDict() if self.taskBuffer is not None else None,
            "ADBStatus": {"Status": "connecting", "Device": ""} if self.initIng else (
                self.automation.operate.getDevicesConnectionStatus() if self.automation is not None else {"Status": "disconnected", "Device": ""}),
        }

    def getSystemInfo(self):
        results = {"CPU": [], "GPU": []}
        physical_devices = GPUtil.getGPUs()
        pc_mem = psutil.virtual_memory()
        for cpu in tf.config.experimental.list_physical_devices('CPU'):
            results["CPU"].append({"name": cpu.name, "maxMemory": self.memoryToString(pc_mem.available), "memoryValue": float(pc_mem.available)})
        index = 0
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            results["GPU"].append({"name": gpu.name, "maxMemory": self.memoryToString(physical_devices[index].memoryFree * 1024 * 1024), "memoryValue": float(physical_devices[index].memoryFree),
                                   "description": physical_devices[index].name})
            index += 1
        return results

    @staticmethod
    def memoryToString(memory: int):
        if memory < 1024:
            return str(memory) + "B"
        elif memory < 1024 * 1024:
            return str(round(memory / 1024, 2)) + "KB"
        elif memory < 1024 * 1024 * 1024:
            return str(round(memory / 1024 / 1024, 2)) + "MB"
        else:
            return str(round(memory / 1024 / 1024 / 1024, 2)) + "GB"


class TaskBuffer:
    def __init__(self, **kwargs):
        self.singleTask: TaskBuffer.SingleTask = kwargs.get("singleTask", TaskBuffer.SingleTask())
        self.continuousTask: TaskBuffer.ContinuousTask = kwargs.get("continuousTask", TaskBuffer.ContinuousTask())
        self.taskType: str = kwargs.get("taskType", "single")

    def load(self, path):
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write('{"singleTask": {}, "continuousTask": {}}')

        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.singleTask = TaskBuffer.SingleTask(**data["singleTask"])
                self.continuousTask = TaskBuffer.ContinuousTask(**data["continuousTask"])
        except Exception as e:
            logger.error(e)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "singleTask": self.singleTask.__dict__,
                "continuousTask": self.continuousTask.getAsDict(),
                "taskType": self.taskType
            }, f)

    def getAsDict(self):
        return {
            "singleTask": self.singleTask.__dict__,
            "continuousTask": self.continuousTask.getAsDict(),
            "taskType": self.taskType
        }

    class SingleTask:
        def __init__(self, **kwargs):
            self.frequency = kwargs.get("frequency", 3)
            self.sanityTimes = kwargs.get("sanityTimes", 0)
            self.useStone = kwargs.get("useStone", False)

    class ContinuousTask:
        def __init__(self, **kwargs):
            self.intervalTime = kwargs.get("intervalTime", 180)
            self.minStartMultiple = kwargs.get("minStartMultiple", 1)
            self.inlineSingleTask: TaskBuffer.SingleTask = TaskBuffer.SingleTask(**kwargs.get("inlineSingleTask", {}))

        def getAsDict(self):
            return {
                "intervalTime": self.intervalTime,
                "minStartMultiple": self.minStartMultiple,
                "inlineSingleTask": self.inlineSingleTask.__dict__
            }
