# -*- coding: utf-8 -*-
"""
The provided Python script integrates the DeepSort algorithm for object tracking with a focus on six specific classes: 'person', 'car', 'bus', 
'truck', 'bicycle', and 'motorbike'. It begins by importing necessary modules and loading configuration settings from a YAML file tailored for DeepSort.
@author: Shiqi Jiang
Created on Wed Jan 24 23:07:20 2024
"""

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")

def deepsort_maker():
    """
    Creates an instance of the DeepSort object using the provided configuration settings.

    Returns:
        DeepSort: An instance of the DeepSort object.
    """
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    return deepsort

def plot_bboxes(image, bboxes, line_thickness=None):
    """
    Plots one bounding box on image img

    Args:
        image (ndarray): The image on which to plot the bounding box.
        bboxes (list): A list of bounding boxes, where each bounding box is represented as a tuple of (x1, y1, x2, y2, class_id, position_id).
        line_thickness (int, optional): The thickness of the lines used to plot the bounding box. If not specified, a default value is chosen based on 
        the size of the image.

    Returns:
        ndarray: The image with the bounding box plotted on it.
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4

    # defining a colour dictionary OBJ_LIST = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorbike']
    # (choose clolors that you want:https://tool.oschina.net/commons?type=3)
    color_dict = {'person': (255, 165, 0), 'car': (238, 221, 130), 'bus': (255, 0, 0),
                  'truck': (139, 117, 0), 'bicycle': (0, 0, 128), 'motorbike': (0, 191, 255)}
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = color_dict.get(cls_id, (0, 255, 0))  # If class is not in color_dict, use green color
    
        # check whether hit line 
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        # tf = max(tl - 1, 1)  # font thickness
        tf = 1
        scale = 6 # font size reduction
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / scale, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{}-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / scale,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        list_pts.clear()            
        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])
    return image

def update(target_detector, image):
    """
    Update the bounding boxes of tracked objects in an image.

    Args:
        target_detector (object): A object detector that returns bounding boxes and class labels for objects in the image.
        image (ndarray): The image on which to perform the object tracking.

    Returns:
        tuple: A tuple containing the updated image and a list of bounding boxes for tracked objects. The bounding boxes are represented as tuples of 
        (x1, y1, x2, y2, class_id, track_id).
    """
    _, bboxes = target_detector.detect(image)

    # Initialize list to store tracked objects
    all_bboxes2draw = []

    # Loop through each class of object and track it
    for idx, name in enumerate(['person', 'car', 'bus', 'truck', 'bicycle', 'motorbike']):
        current_bbox= [i for i in bboxes if i[4] == name]

        # Initialize lists to store bounding box coordinates and confidence scores
        bbox_xywh = []
        bboxes2draw = []
        confs = []

        # Check if any objects of the current class were detected
        if len(current_bbox):
            # Loop through each detected object and format the bounding box coordinates and confidence scores for DeepSort
            for x1, y1, x2, y2, cls_id, conf in current_bbox:
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)

            # Convert the lists of bounding box coordinates and confidence scores to PyTorch Tensors
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # Pass the formatted bounding box coordinates and confidence scores to DeepSort for tracking
            outputs = deepsort_list[idx].update(xywhs, confss, image)

            # Loop through the outputs from DeepSort and draw the tracked bounding boxes on the image
            for value in list(outputs):
                x1,y1,x2,y2,track_id = value
                bboxes2draw.append((x1, y1, x2, y2, cls_id, track_id))
                all_bboxes2draw.append((x1, y1, x2, y2, cls_id, track_id))
            image = plot_bboxes(image, bboxes2draw)

    return image, all_bboxes2draw

# Since six classes for tracking are needed, each with its ID, six trackers are created
deepsort_list = [deepsort_maker() for i in range(6)]