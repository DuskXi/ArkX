import time

from loguru import logger

from .automation import Automation, Task


class ContinuousTask:
    def __init__(self, automation: Automation, intervalTime: int, minStartMultiple: int = 2):
        self.automation = automation
        self.intervalTime = intervalTime
        self.minStartMultiple = minStartMultiple
        self.task = automation.task.copy()
        self.isRun = True
        self.status = Status.Stopped
        self.abort = False

    def sleep(self, timeLong):
        startTime = time.time()
        while time.time() - startTime < timeLong:
            time.sleep(1)
            if self.abort:
                break

    def run(self, callback=None):
        self.isRun = True
        while self.isRun:
            self.status = Status.Running
            self.automation.gotoFightSelection()
            sanity, sanityMax = self.automation.operate.getSanity()
            levelCost = self.automation.operate.getLevelSanity(self.automation.labelsName.MainStart)
            if sanity < levelCost * self.minStartMultiple and sanity < sanityMax - 10:
                logger.info(f"当前理智: {sanity}/{sanityMax}")
                logger.info(f"不满足 {self.minStartMultiple}倍于{levelCost}的启动条件")
                d = (levelCost * self.minStartMultiple) - sanity
                gapTime = d * 5 * 60
                logger.info(f"进入休眠, 等待 {gapTime} 秒")
                self.status = Status.Sleeping
                self.sleep(gapTime)
                continue
            self.automation.reset(self.task.copy())
            self.status = Status.RunningTask
            self.automation.mainLoop()
            logger.info("当前阶段任务完成, 进入休眠: {}s".format(self.intervalTime))
            self.status = Status.Sleeping
            self.sleep(self.intervalTime)
            if self.abort:
                break

        self.status = Status.Stopped

        logger.info("持续性任务结束运行")
        if self.abort:
            logger.warning("持续性任务被强制终止")
            self.abort = False
        if callback is not None:
            callback()

    def stop(self):
        self.abort = True
        self.isRun = False
        if self.automation.isRun:
            self.automation.stop()


class Status:
    Stopped = "Stopped"
    RunningTask = "RunningTask"
    Running = "Running"
    Sleeping = "Sleeping"
