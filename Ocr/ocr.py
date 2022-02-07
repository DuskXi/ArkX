import time

import numpy as np
from paddleocr import PaddleOCR

from Performance import recoder


# 如果有莫名其妙的print， 请到 predict_system.py 里面删除
class PDOcr:
    def __init__(self, models, use_gpu=False, gpu_mem=2048) -> None:
        (det, cls, rec) = models
        self.ocr = PaddleOCR(det_model_dir=det, rec_model_dir=rec, cls_model_dir=cls, use_angle_cls=True, gpu_mem=gpu_mem, use_gpu=use_gpu)

    def getTextsAndPosition(self, img: np.ndarray):
        start = time.time()
        results = self.ocr.ocr(img)
        timeLong = time.time() - start
        sumPercentage = 0
        for result in results:
            result[1][1] = float(result[1][1])
            sumPercentage += result[1][1]
        if len(results) > 0:
            recoder.Recoder.recordOcr(timeLong, sumPercentage / len(results))
        return results

    def detectByArea(self, img: np.ndarray, xMin, xMax, yMin, yMax):
        results = self.getTextsAndPosition(img)
        data = []
        for result in results:
            minX = result[0][0][0]
            maxX = 0
            for x in np.array(result[0])[:, 0]:
                if x < minX:
                    minX = x
                if x > maxX:
                    maxX = x
            minY = result[0][0][1]
            maxY = 0
            for y in np.array(result[0])[:, 1]:
                if y < minY:
                    minY = y
                if y > maxY:
                    maxY = y
            if (xMax >= maxX > xMin or xMin <= minX < xMax) and (yMax >= maxY > yMin or yMin <= minY < yMax):
                data.append({"text": result[1][0], "position": result[0]})
        return data
