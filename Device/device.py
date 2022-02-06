from time import sleep

from loguru import logger

from .viewer import AndroidViewer


class Device:
    android = None
    isRun = True
    latestScreen = None
    isRead = None

    def __init__(self, adb_path):
        if adb_path != "-1":
            logger.info("initialize AndroidViewer")
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
