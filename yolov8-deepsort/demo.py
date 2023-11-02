import os.path
import time

import numpy as np
import objtracker
from objdetector import Detector
import cv2

VIDEO_PATH = './video/hauptmensa_1.mp4'
RESULT_PATH = './output/hauptmensa_1_manu.mp4'
TXT_PATH = "./output/det_hauptmenda_manu.txt"

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

def center_pos(x1,y1,x2,y2,cls_id,track_id):
    center_x = int((x2-x1)/2+x1)
    center_y = int((y2-y1)/2+y1)
    return (center_x,center_y,cls_id,track_id)

if __name__ == '__main__':
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    detector = Detector()
    capture = cv2.VideoCapture(VIDEO_PATH)
    videoWriter = None
    fps = int(capture.get(5))
    print('fps:', fps)

    # Dictionary to store the trail points of each object (choose clolors that you want:https://tool.oschina.net/commons?type=3)
    color_dict = {'person': (255, 165, 0), 'car': (238, 221, 130), 'bus': (255, 0, 0),
                  'truck': (139, 117, 0), 'bicycle': (0, 0, 128), 'motorbike': (0, 191, 255)}

    # 定义跟踪轨迹字典
    track_dict = dict()
    t = 0
    if os.path.exists(TXT_PATH):
        os.remove(TXT_PATH)
    while True:
        t+=1
        _, im = capture.read()
        if im is None:
            break
        # if t % 10 != 0: # 跳帧
        #     continue
        detections = Detections()
        output_image_frame, list_bboxs = objtracker.update(detector, im)
        for i in list_bboxs:
            # 没有cls_id
            txt = f"{t}, {i[5]}, {round(i[0],2)}, {round(i[1],2)}, {round(i[2]-i[0],2)}, {round(i[3]-i[1],2)}, 1, -1, -1, -1\n"
            # # 有cls_id
            # txt = f"{t}, {i[5]}, {i[4]}, {round(i[0],2)}, {round(i[1],2)}, {round(i[2]-i[0],2)}, {round(i[3]-i[1],2)}, 1, -1, -1, -1\n"
            with open(TXT_PATH,"a") as f:
                f.write(txt)
        # x1，y1, x2, y2, cls_id, track_id -> centerx,center_y, cls_id, track_id
        list_box = [center_pos(i[0],i[1],i[2],i[3],i[4],i[5]) for i in list_bboxs]

        # 遍历当前帧识别内容
        for box in list_box:

            # 查到跟踪字典里有没有名为 cls_id_track_id的key
            if track_dict.get(f"{box[2]}_{box[3]}") == None: # 没有则更新这个key
                track_dict.update({f"{box[2]}_{box[3]}":[]})
                track_dict[f"{box[2]}_{box[3]}"].append([box[0],box[1]])
            else: # 有则向key的value里增加一组最新点位坐标
                track_dict[f"{box[2]}_{box[3]}"].append([box[0], box[1]])

        # 定义删除消失的track_id
        del_key = []
        # 遍历跟踪字典
        for key,value in track_dict.items():
            cls_id, track_id = key.split("_")
            clear_flag = 1 # 定义删除标识位

            # 如果在字典中查到当前帧识别到对应的cls_id及track_id则此key保留
            for box in list_box:
                if box[2] == cls_id and str(box[3]) == track_id:
                    clear_flag = 0
            if clear_flag == 1: # 如果flag没有被赋值为0证明此条key在当前帧没被识别到，删除列表增加此key
                del_key.append(key)

        # 循环当前需要删除的key对字典里的key进行删除
        for key in del_key:
            del track_dict[key]

        # 遍历字典对图像进行轨迹画图
        for key, value in track_dict.items():
            cls_id, track_id = key.split("_")
            color = color_dict[cls_id]
            for i in range(len(value)-1):
                x1,y1 = value[i][0],value[i][1]
                x2,y2 = value[i+1][0], value[i+1][1]
                cv2.line(output_image_frame,(x1,y1),(x2,y2),color,2)
        # 假设detector.detect返回一个元组，其中第一个元素是图像，第二个元素是一个列表，列表中的每个元素是一个元组，元组的最后一个元素是类别ID
        # image, detections = detector.detect(im)
        # for detection in detections:
        #     cls_id = detection[-1]  # 假设类别ID是元组的最后一个元素
        #     output_image_frame, list_bboxs = objtracker.update(detector, im, cls_id)

        # for item_bbox in list_bboxs:
        #     x1, y1, x2, y2, _, track_id = item_bbox
        #     detections.add((x1, y1, x2, y2), None, None, track_id)
        #
        # # Add the current object's position to the trail
        # for xyxy, _, _, track_id in detections.detections:
        #     x1, y1, x2, y2 = xyxy
        #     center = Point(x=(x1+x2)/2, y=(y1+y2)/2)
        #     if track_id in object_trails:
        #         object_trails[track_id].append((center.x, center.y))
        #     else:
        #         object_trails[track_id] = [(center.x, center.y)]
        #
        # # Draw the trail for each object
        # trail_colors = [(255, 0, 255)] * len(object_trails)  # Red color for all trails
        # draw_trail(output_image_frame, list(object_trails.values()), trail_colors)
        #
        # # Remove trails of objects that are not detected in the current frame
        # for tracker_id in list(object_trails.keys()):
        #     if tracker_id not in [item[3] for item in detections.detections]:
        #         object_trails.pop(tracker_id)
        #
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

        videoWriter.write(output_image_frame)
        cv2.imshow('Demo', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
