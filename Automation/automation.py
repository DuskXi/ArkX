import re
import time

from loguru import logger

from .operate import Operate
from Performance import recoder


class Automation:
    def __init__(self, operate: Operate, labelsName, task):
        self.operate = operate
        self.screen = ""
        self.isRun = True
        self.labelsName = LabelsName(**labelsName)
        self.screenStatus = ScreenStatus()
        self.task: Task = task
        # 单位: 秒
        self.intervalTime = 0.5
        self.workingIntervalTime = 3
        self.waitingTimeOut = 5
        self.levelInfo = {}
        self.screenInfo = {}
        self.progress = None
        self.abort = False
        self.inited = False

    def loadGame(self, device_name=None):
        self.operate.loadDevices(device_name=device_name)
        self.operate.waitForDevices()
        self.inited = True

    def releaseGame(self):
        self.operate.releaseDevices()

    def updateScreen(self):
        screen = self.operate.getGameStatus()
        if self.screen != screen:
            logger.debug(f"游戏界面变更为: {screen}")
        self.screen = screen

    def stepSleep(self):
        time.sleep(self.intervalTime)

    def workingSleep(self):
        time.sleep(self.workingIntervalTime)

    def mainLoop(self, callback=None):
        self.updateScreen()
        self.operate.getObjectDataMain()
        self.levelInfo["cost"] = self.operate.getLevelSanity(self.labelsName.MainStart)
        logger.info(f"任务开始, 关卡理智消耗: {self.levelInfo['cost']}")
        while self.isRun:
            self.updateScreen()
            if self.screen == self.screenStatus.LevelPrepare:
                self.updateScreenInfo()
                if self.task.frequency > 0:
                    if not self.keepProxy():
                        self.stepSleep()
                        continue
                    if self.levelInfo["cost"] is not None:
                        sanity = self.operate.getSanity()[0]
                        if sanity is not None:
                            if sanity < self.levelInfo["cost"] and self.task.useSanity < 1:
                                logger.info(f"理智不足, 当前理智: {sanity}, 花费理智: {self.levelInfo['cost']}")
                                self.isRun = False
                                break
                            else:
                                if self.task.useSanity < 1:
                                    logger.info(f"开始新一轮循环, 当前理智: {sanity}, 即将花费理智: {self.levelInfo['cost']}")
                                else:
                                    logger.info(f"开始新一轮循环, 当前理智: {sanity}, 即将花费理智: {self.levelInfo['cost']}, 准备进入添加理智进程")

                    box = self.operate.getBoxMain(self.labelsName.MainStart)
                    if box is not None:
                        self.operate.clickBox(box["position"])
                    self.waitFor([self.screenStatus.Team, self.screenStatus.AddSanity])
                    continue
                else:
                    self.isRun = False
                    break

            if self.screen == self.screenStatus.Team:
                box = self.operate.getBoxMain(self.labelsName.TeamStart)

                if recoder.Recoder.debug:
                    logger.debug("检测到debug模式, 开始睡眠")
                    start = time.time()
                    while time.time() - start < recoder.Recoder.debugSleepingTime:
                        time.sleep(1)

                if box is not None:
                    self.operate.clickBox(box["position"])
                    self.waitFor([self.screenStatus.Working])
                    self.task.frequency -= 1
                self.stepSleep()
                continue

            if self.screen == self.screenStatus.Working:
                self.progress = self.operate.getRunningInfoInWorking()
                self.stepSleep()
                self.workingSleep()
                continue

            if self.screen == self.screenStatus.Ending:
                self.progress = None
                resolution = self.operate.getResolution()
                self.operate.click(resolution["x"] - (resolution["x"] / 10), resolution["y"] / 4)
                self.waitFor([self.screenStatus.LevelPrepare])
                self.stepSleep()
                continue

            if self.screen == self.screenStatus.AddSanity:
                result = self.addSanity()
                logger.debug(f"理智已添加: [{result}]")
                self.stepSleep()
                self.workingSleep()
                continue

        logger.debug("任务结束" if self.screen in [self.screenStatus.LevelPrepare, self.screenStatus.Ending, self.screenStatus.AddSanity] else "任务中断")
        if self.abort:
            logger.warning("任务被强制中断")
            self.abort = False
            return
        if callback is not None:
            callback()

    def stop(self):
        self.abort = True
        self.isRun = False

    def waitFor(self, screenStatus: list):
        start = time.time()
        while time.time() - start < self.waitingTimeOut:
            self.updateScreen()
            if self.screen in screenStatus:
                return True
            time.sleep(self.intervalTime / 2)
        return False

    def getLevelSanity(self):
        return self.operate.getLevelSanity(self.labelsName.MainStart)

    def updateScreenInfo(self):
        if self.inited:
            self.updateScreen()
            self.screenInfo = {
                "screen": self.screen,
                "sanity": self.operate.getSanity(),
                "levelSanity": self.getLevelSanity(),
                "proxy": self.operate.getProxyStatus(),
            }

    def getScreenInfo(self):
        if self.inited:
            return self.screenInfo
        return None

    def addSanity(self):
        if self.task.useSanity > 0:
            texts = self.operate.runOcr()
            for text in texts:
                if "是否花费" in text["text"]:
                    if "源石" in text["text"]:
                        if self.task.enableStone:
                            box = self.operate.getBoxSanity(self.labelsName.SanityConfirm)
                            self.operate.clickBox(box["position"])
                            self.waitFor([self.screenStatus.LevelPrepare])
                            self.task.useSanity -= 1
                            return re.search("[0-9]+", text["text"])[0]
                    else:
                        box = self.operate.getBoxSanity(self.labelsName.SanityConfirm)
                        self.operate.clickBox(box["position"])
                        self.waitFor([self.screenStatus.LevelPrepare])
                        self.task.useSanity -= 1
                        return re.search("[0-9]+", text["text"])[0]

    def keepProxy(self):
        result = self.operate.getProxyStatus()
        if result is None:
            return False
        if not result:
            box = self.operate.getBoxMain(self.labelsName.Proxy)
            if box is not None:
                self.operate.clickBox(box["position"])
                self.stepSleep()
            else:
                raise BoxNotFoundException("Proxy Not Found!")
        return True

    def gotoFightSelection(self):
        self.updateScreen()
        if self.screen is not self.screenStatus.LevelPrepare:
            resolution = self.operate.getResolution()
            self.operate.click(resolution["x"] - 50, resolution["y"] / 4)
            self.waitFor([self.screenStatus.LevelPrepare])

    def reset(self, task):
        self.task = task
        self.isRun = True


class LabelsName:
    def __init__(self, **entries):
        self.Sanity = "Sanity"
        self.Proxy = "Proxy"
        self.MainStart = "MainStart"
        self.TeamStart = "TeamStart"
        self.SanityConfirm = "Confirm"
        self.__dict__.update(entries)


class ScreenStatus:
    def __init__(self):
        self.AddSanity = "AddSanity"
        self.LevelList = "LevelList"
        self.LevelPrepare = "LevelPrepare"
        self.Main = "Main"
        self.Team = "Team"
        self.Ending = "Ending"
        self.Working = "Working"


class Task:
    def __init__(self, frequency: int, useSanity: int, enableStone: bool = False):
        self.frequency = frequency
        self.useSanity = useSanity
        self.enableStone = enableStone

    def copy(self):
        return Task(self.frequency, self.useSanity, self.enableStone)

    def getTaskInfo(self):
        return {"frequency": self.frequency, "useSanity": self.useSanity, "enableStone": self.enableStone}


class BoxNotFoundException(Exception):
    pass
