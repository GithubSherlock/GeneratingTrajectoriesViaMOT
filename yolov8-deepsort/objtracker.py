from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")

# 追踪器创造函数
def deepsort_maker():
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    return deepsort

# 由于需要六类追踪，每类都有各自的id，所以要创建六个追踪器
deepsort_list = [deepsort_maker() for i in range(6)]


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4

    # 定义颜色字典 OBJ_LIST = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorbike'] (choose clolors that you want:https://tool.oschina.net/commons?type=3)
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
        scale = 6 # 字体缩小倍数
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

        ndarray_pts = np.array(list_pts, np.int32)
        # cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        #list_pts.clear()
    return image

def update(target_detector, image):
        _, bboxes = target_detector.detect(image)

        # 初始化多标签坐标结果return list
        all_bboxes2draw = []

        # 遍历6种识别目标进行画图及deepsort计算
        for idx, name in enumerate(['person', 'car', 'bus', 'truck', 'bicycle', 'motorbike']):
            current_bbox= [i for i in bboxes if i[4] == name]
            bbox_xywh = []
            bboxes2draw = []
            confs = []
            if len(current_bbox):
                # Adapt detections to deep sort input format
                for x1, y1, x2, y2, cls_id, conf in current_bbox:
                    obj = [
                        int((x1+x2)/2), int((y1+y2)/2),
                        x2-x1, y2-y1
                    ]
                    bbox_xywh.append(obj)
                    confs.append(conf)
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                # 利用给定编号追踪器进行追踪更新
                outputs = deepsort_list[idx].update(xywhs, confss, image)
                for value in list(outputs):
                    x1,y1,x2,y2,track_id = value
                    bboxes2draw.append((x1, y1, x2, y2, cls_id, track_id))
                    all_bboxes2draw.append((x1, y1, x2, y2, cls_id, track_id))
                image = plot_bboxes(image, bboxes2draw)

        return image, all_bboxes2draw
