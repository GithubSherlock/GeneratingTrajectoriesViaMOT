# -*- coding: utf-8 -*-
"""
The provided Python script is designed for object detection using a YOLO model from the Ultralytics library. It defines a list of objects to detect, 
such as 'person', 'car', 'bus', 'truck', 'bicycle', and 'motorbike', and specifies a path to the pre-trained model weights.
@author: Shiqi Jiang
Created on Wed Jan 24 23:06:20 2024
"""

import torch
from ultralytics import YOLO

# Constant Control Board
OBJ_LIST = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorbike']
DETECTOR_PATH = 'weights/sportscheck353.pt' # yolov8s, yolov8n, yolov8m, yolov8l, yolov8x, mensa3, mensa300, sportscheck300, mensa500, mensa200 sportscheck353
# Constant Control Board

class baseDet(object):
    """
    The baseDet class provides default values for the image size, confidence threshold, and IOU threshold used in object detection.

    Args:
        img_size (int): The default image size used for preprocessing and inference. Defaults to 640.
        conf (float): The default confidence threshold used to filter detections. Defaults to 0.5.
        iou (float): The default intersection-over-union threshold used to filter overlapping bounding boxes. Defaults to 0.6.
    """
    def __init__(self):
        """
        Initialize the baseDet class with default values for the image size, confidence threshold, and IOU threshold.
        """
        self.img_size = 640
        self.conf = 0.60
        self.iou = 0.50

    def init_model(self):
        """
        This method initializes the model.
        """
        raise EOFError("Undefined model type.")

    def preprocess(self):
        """
        This method preprocesses the input data.
        """
        raise EOFError("Undefined model type.")

    def detect(self):
        """
        This method performs the detection.
        """
        raise EOFError("Undefined model type.")


class Detector(baseDet):
    """
    The Detector class provides a high-level interface for object detection using a YOLO model. It defines a list of objects to detect, 
    such as 'person', 'car', 'bus', 'truck', 'bicycle', and 'motorbike', and specifies a path to the pre-trained model weights.

    Args:
        img_size (int): The default image size used for preprocessing and inference. Defaults to 640.
        conf (float): The default confidence threshold used to filter detections. Defaults to 0.5.
        iou (float): The default intersection-over-union threshold used to filter overlapping bounding boxes. Defaults to 0.6.
    """
    def __init__(self):
        """
        The __init__ method initializes the baseDet class.
        """
        super(Detector, self).__init__()
        self.init_model()

    def init_model(self):
        """
        initializes the YOLO model using the specified weights and determines the computation device based on GPU availability.

        Returns:
            None
        """
        self.weights = DETECTOR_PATH
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.weights)
        self.m = self.model
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, im):
        """
        Runs object detection on an input image.

        Args:
            im (numpy.ndarray): Input image as a numpy array.

        Returns:
            tuple: A tuple containing the input image and a list of tuples containing the bounding boxes, labels, and confidence scores of the detected objects.

        """
        res = self.model.predict(im, imgsz=self.img_size, conf=self.conf,
                                     iou=self.iou, device=self.device)
                    
        detected_boxes = res[0].boxes
        pred_boxes = []
        for box in detected_boxes:
            xyxy = box.xyxy.cpu() 
            #print(xyxy)
            confidence = box.conf.cpu() 
            class_id = box.cls  # get the class id
            class_id_cpu = class_id.cpu()  # move the value to CPU
            class_id_int = int(class_id_cpu.item())  # convert to integer
            lbl = self.names[class_id_int]
            if not lbl in OBJ_LIST:
                continue
            x1, y1, x2, y2 = xyxy[0].numpy()
            pred_boxes.append(
                 (x1, y1, x2, y2, lbl, confidence))
        return im, pred_boxes
