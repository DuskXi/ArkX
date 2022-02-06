import pathlib
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from loguru import logger
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from tensorflow import keras
from typing import Dict

from Performance import recoder


def set_gpu_memory(memory=3072):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.warning("set: " + str(gpus) + " Memory=" + str(memory))
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])


def set_force_use_cpu():
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')
    tf.config.set_visible_devices([], 'GPU')
    logger.warning("setForceUseCPU: " + str(my_devices))


def set_dynamic_gpu_memory_usage():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.warning("GPU 被设置为动态显存分配")


# noinspection PyTypeChecker
class ConvolutionalNeuralNetworkCore:
    """
    卷积神经网络核心类

    简称定义:
    ODModel: ModelObjectDetection(对象检测模型)
    ICModel: ModelImageClassification(图像分类模型)
    """

    def __init__(self):
        self.dictODModel: Dict[str, ModelObjectDetection] = {}
        self.dictICModel: Dict[str, ModelImageClassification] = {}

    def loadODModel(self, name, path, pbtxt):
        """
        加载对象检测模型
        :param name: 模型id
        :param path: 模型路径
        :param pbtxt: 模型pbtxt(index对应objectName的文件)文件路径
        :return: None
        """
        model = ModelObjectDetection()
        model.loadModel(name, path, pbtxt)
        self.dictODModel[name] = model

    def detectObject(self, name, image):
        """
        对象检测
        :param name: 模型id
        :param image: 图像(ndarray)
        :return: 对象检测结果
        """
        return self.dictODModel[name].detectObject(image)

    def drawBox(self, name, image):
        """
        在传入的图像中绘制缓存中的对象检测结果
        :param name: 模型id
        :param image: 图像(ndarray)
        :return: None
        """
        self.dictODModel[name].drawBox(image)

    def getBoxData(self, name, im_width, im_height) -> dict:
        """
        返回经过解析后的box数据
        :param name: 模型id
        :param im_width: 图像宽度
        :param im_height: 图像高度
        :return: box数据
        """
        return self.dictODModel[name].getBoxData(im_width, im_height)

    def loadICModel(self, name, path, classes):
        """
        加载图像分类模型
        :param name: 模型id
        :param path: 模型路径
        :param classes: id与label名的对应关系
        :return: None
        """
        model = ModelImageClassification()
        model.loadModel(name, path, classes)
        self.dictICModel[name] = model

    def classifyImages(self, name, image) -> dict:
        """
        对图像进行分类
        :param name: 模型id
        :param image: 图像(ndarray)
        :return: 原始分类数据
        """
        image_np = np.array(Image.fromarray(image.copy()).resize((960, 540)))
        return self.dictICModel[name].classifyImages(np.array([image_np]))

    def getClassification(self, name, image) -> [str]:
        """
        返回经过解析后的图像分类结果
        :param name: 模型id
        :param image: 图像(ndarray)
        :return: 可能性最大的图像类型
        """
        image_np = np.array(Image.fromarray(image.copy()).resize((960, 540)))
        return self.dictICModel[name].getClassification(np.array([image_np]))

    def getClassificationFromBuffer(self, name) -> [str]:
        """
        从缓存中返回经过解析后的图像分类结果
        :param name: 模型id
        :return: 可能性最大的图像类型
        """
        return self.dictICModel[name].getClassificationFromBuffer()


class ModelImageClassification:
    model = None
    name = None
    classes = None
    predictions = None
    bufferAvailable = False

    def loadModel(self, name, path, classes):
        model = keras.models.load_model(path)
        self.model = model
        self.name = name
        self.classes = classes

    def classifyImages(self, images):
        predictions = self.model.predict(images)
        self.predictions = predictions
        self.bufferAvailable = True
        return predictions

    def getClassification(self, images) -> [str]:
        start = time.time()
        predictions = self.classifyImages(images)
        timeLong = time.time() - start
        classes = []
        maxValue = 0
        for prediction in predictions:
            maxAccuracy = 0
            for i in range(len(prediction)):
                if prediction[i] > prediction[maxAccuracy]:
                    maxAccuracy = i
            maxValue = prediction[maxAccuracy]
            classes.append(self.classes[maxAccuracy])
        recoder.Recoder.recordClassifyModel(self.name, timeLong, maxValue)
        return classes

    def getClassificationFromBuffer(self) -> [str]:
        predictions = self.predictions
        classes = []
        for prediction in predictions:
            maxAccuracy = 0
            for i in range(len(prediction)):
                if prediction[i] > prediction[maxAccuracy]:
                    maxAccuracy = i
            classes.append(self.classes[maxAccuracy])
        return classes


class ModelObjectDetection:
    model = None
    name = None
    pbtxt = None
    category_index = None
    output_dict = None
    bufferAvailable = False

    def loadModel(self, name, path, pbtxt):
        model_dir = path
        model_dir = pathlib.Path(model_dir) / "saved_model"
        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']
        self.model = model
        self.name = name
        self.pbtxt = pbtxt
        self.category_index = label_map_util.create_category_index_from_labelmap(pbtxt, use_display_name=True)

    def run_inference_for_single_image(self, image):
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        output_dict = self.model(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        if 'detection_masks' in output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        return output_dict

    def detectObject(self, image):
        start = time.time()
        output_dict = self.run_inference_for_single_image(image)

        timeLong = time.time() - start
        performanceData = {}
        for i in range(len(output_dict['detection_classes'])):
            performanceData[self.category_index[output_dict['detection_classes'][i]]["name"]] = output_dict['detection_scores'][i]
        recoder.Recoder.recordObjectModel(self.name, timeLong, performanceData)

        self.output_dict = output_dict
        self.bufferAvailable = True
        return output_dict

    def drawBox(self, image):
        output_dict = self.output_dict
        vis_util.visualize_boxes_and_labels_on_image_array(image, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], self.category_index,
                                                           instance_masks=output_dict.get('detection_masks_reframed', None), use_normalized_coordinates=True, line_thickness=8)

    def getBoxData(self, im_width, im_height) -> dict:
        output_dict = self.output_dict
        boxes = np.squeeze(output_dict['detection_boxes'])
        scores = np.squeeze(output_dict['detection_scores'])
        classes = np.squeeze(output_dict['detection_classes'])
        min_score_thresh = 0.50
        bboxes = boxes[scores > min_score_thresh]
        names = classes[scores > min_score_thresh]
        percents = scores[scores > min_score_thresh]
        final_box = []
        labels = []
        for box, name in zip(bboxes, names):
            yMin, xMin, yMax, xMax = box
            final_box.append([xMin * im_width, xMax * im_width,
                              yMin * im_height, yMax * im_height])
            labels.append(self.category_index[name])
        return {"boxes": final_box, "labels": labels, "percents": percents.tolist()}
