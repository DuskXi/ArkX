import re
from time import sleep

from loguru import logger
import subprocess

from .viewer import AndroidViewer


class Device:
    android = None
    isRun = True
    latestScreen = None
    isRead = None

    def __init__(self, adb_path, device_name=None):
        self.device_name = ""
        if adb_path != "-1":
            logger.info("initialize AndroidViewer")
            if device_name is not None:
                self.android = AndroidViewer(adb_path=adb_path, bitrate=5632000, multipleDevices=True, deviceADBName=device_name)
                self.device_name = device_name
            else:
                self.android = AndroidViewer(adb_path=adb_path, bitrate=5632000)
            self.isRun = True
        else:
            logger.warning("initialize AndroidViewer Test Mode")

    def keepScreenRefresh(self):
        logger.debug("keepScreenRefresh thread is started.")
        while self.isRun:
            frames = self.android.get_next_frames()
            if frames is None:
                continue
            self.latestScreen = frames[frames.__len__() - 1]
            self.isRead = False
        logger.debug("keepScreenRefresh thread has exited.")

    def readMirror(self):
        return self.latestScreen.copy()

    def readScreen(self):
        self.isRead = True
        return self.latestScreen

    def waitNewScreen(self):
        while self.isRead:
            sleep(0.001)
        self.isRead = True
        return self.latestScreen

    def synchronizeGetScreen(self):
        self.android.get_next_frames()

    # noinspection PyBroadException
    def click(self, x, y):
        try:
            self.android.click(x, y)
        except Exception:
            pass

    def getDeviceName(self):
        return self.android.device_name


class DeviceManager:
    def __init__(self, adb_path):
        self.adb_path = adb_path

    def getADBDevices(self):
        devices = []

        adb = subprocess.Popen([self.adb_path, 'devices'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        adb.wait()
        result, error = adb.communicate()
        startLineFound = False
        for line in result.splitlines():
            content = line.decode('utf-8')
            if content.startswith('List of devices attached'):
                startLineFound = True
                continue
            if startLineFound:
                if re.search(r'[0-9a-z:\.]+\s(device|offline)', content):
                    device = re.search(r'^[0-9a-z:\.]+', content)[0]
                    status = re.search(r'(device|offline)', content)[0]
                    devices.append([device, status])

        return devices
