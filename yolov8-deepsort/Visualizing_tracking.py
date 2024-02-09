# -*- coding: utf-8 -*-
"""
The provided script is designed for object tracking in video files, specifically targeting a sports context. It begins by setting up 
paths for the video input, output, and a text file to log detection results. The script defines a `Point` class for coordinates and a 
`Detections` class to manage detected objects. A key function, `draw_trail`, is used to draw trails behind tracked objects, enhancing visual 
tracking over video frames. During video processing, for each frame, detected objects are logged into a text file with their bounding box 
coordinates and class identifiers. The `center_pos` function calculates the center position of each detected object for tracking purposes. 
The script maintains a tracking dictionary, updating it with new detections or removing those no longer present. The script also includes 
logic to draw tracking lines between consecutive positions of each object, using predefined colors for visual distinction. Finally, the 
processed frames are written to an output video file, and the results can be viewed in real-time through an OpenCV window.
@author: Shiqi Jiang
Created on Wed Jan 24 23:47:00 2024
"""

import os.path
import time
import numpy as np
import objtracker
from objdetector import Detector
import cv2

# Constant Control Board
VIDEO_PATH = './video/sportscheck.mp4'
RESULT_PATH = './output/sportscheck.mp4'
TXT_PATH = './output/det_sportscheck.txt'
# Constant Control Board

class Point:
    """
    A class to represent a point in a 2D plane.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.

    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    """
    A class to store and manage detections.

    Attributes:
        detections (list): A list of tuples containing detection information. Each tuple contains
            (xyxy, confidence, class_id, tracker_id), where xyxy are the bounding box coordinates,
            confidence is the detection confidence, class_id is the object class ID, and tracker_id is
            the object tracker ID.
    """
    def __init__(self):
        """
        Initialize the Detections class.
        """
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        """
        Add a detection to the list of detections.

        Args:
            xyxy (list): The bounding box coordinates of the detection.
            confidence (float): The detection confidence.
            class_id (int): The object class ID.
            tracker_id (int): The object tracker ID.
        """
        self.detections.append((xyxy, confidence, class_id, tracker_id))
    
def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    """
    Draws a trail of points on an image.

    Args:
        output_image_frame (numpy.ndarray): The image on which to draw the trail.
        trail_points (list): A list of lists, where each sublist represents a point
            trail.
        trail_color (list): A list of colors to use for each point trail.
        trail_length (int, optional): The maximum length of each point trail.
            Points beyond this length will be removed. Defaults to 50.

    Returns:
        numpy.ndarray: The image with the trail drawn on it.
    """
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

def center_pos(x1,y1,x2,y2,cls_id,track_id):
    """
    Calculates the center position of a bounding box.

    Args:
        x1 (float): The x-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the top-left corner of the bounding box.
        x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (float): The y-coordinate of the bottom-right corner of the bounding box.
        cls_id (int): The object class ID.
        track_id (int): The object tracker ID.

    Returns:
        tuple: The center position of the bounding box as (center_x, center_y, cls_id, track_id).
    """
    center_x = int((x2-x1)/2+x1)
    center_y = int(y2) # (y2-y1)/2+y1 for cy of bbox
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
    print('fps:', fps, '\nWidth:', width, '\nHeight:', height)

    # Dictionary to store the trail points of each object (choose clolors that you want:https://tool.oschina.net/commons?type=3)
    color_dict = {'person': (255, 165, 0), 'car': (238, 221, 130), 'bus': (255, 0, 0),
                  'truck': (139, 117, 0), 'bicycle': (0, 0, 128), 'motorbike': (0, 191, 255)}

    # Define the tracking trajectory dictionary
    track_dict = dict()
    t = 0
    if os.path.exists(TXT_PATH):
        os.remove(TXT_PATH)
    while True:
        t+=1
        _, im = capture.read()
        if im is None:
            break
        detections = Detections()
        output_image_frame, list_bboxs = objtracker.update(detector, im)
        for i in list_bboxs:
            # without cls_id
            txt = f"{t},{i[5]},{round(i[0],2)},{round(i[1],2)},{round(i[2]-i[0],2)},{round(i[3]-i[1],2)},1,-1,-1,-1\n"
            # with cls_id
            # txt = f"{t}, {i[5]}, {i[4]}, {round(i[0],2)}, {round(i[1],2)}, {round(i[2]-i[0],2)}, {round(i[3]-i[1],2)}, 1, -1, -1, -1\n"
            with open(TXT_PATH,"a") as f:
                f.write(txt)
        # x1，y1, x2, y2, cls_id, track_id -> centerx,center_y, cls_id, track_id
        list_box = [center_pos(i[0],i[1],i[2],i[3],i[4],i[5]) for i in list_bboxs]

        # Iterate through the contents of the currently recognised frame
        for box in list_box:
            # Find out if there is a key with the names cls_id and track_id in the tracking dictionary
            if track_dict.get(f"{box[2]}_{box[3]}") == None: # If not, update the key
                track_dict.update({f"{box[2]}_{box[3]}":[]})
                track_dict[f"{box[2]}_{box[3]}"].append([box[0],box[1]])
            else: # If yes, add the latest coordinate to the track_dict of the key
                track_dict[f"{box[2]}_{box[3]}"].append([box[0], box[1]])

        # Define a list used to store the keys of the objects that need to be removed from the tracking dictionary
        del_key = []
        """
        The code loops over the list of bounding boxes returned by the object detector, and checks if the class ID and tracker ID of 
        the current bounding box match the class ID and tracker ID of the current key. If they do, the code sets a flag called clear_flag 
        to 0, indicating that the object should be kept in the tracking dictionary. If the flag is not set to 0, it means that the object 
        was not detected in the current frame, and therefore it should be removed from the tracking dictionary. The code adds the key to 
        the del_key list, indicating that it needs to be removed.
        Finally, the code loops over the del_key list and removes the corresponding keys from the tracking dictionary. This ensures that 
        objects that were detected in previous frames but are no longer present in the current frame are removed from the tracking dictionary.
        """
        for key,value in track_dict.items():
            cls_id, track_id = key.split("_")
            clear_flag = 1 # Define the deletion identity bit
            for box in list_box:
                if box[2] == cls_id and str(box[3]) == track_id:
                    clear_flag = 0
            if clear_flag == 1: # If the flag is not set to 0, it should be removed from the tracking dictionary
                del_key.append(key)

        # Delete the current key in the dictionary by looping over it.
        for key in del_key:
            del track_dict[key]

        # Iterate over the dictionary to draw trajectories on tracked objects
        for key, value in track_dict.items():
            cls_id, track_id = key.split("_")
            color = color_dict[cls_id]
            for i in range(len(value)-1):
                x1,y1 = value[i][0],value[i][1]
                x2,y2 = value[i+1][0], value[i+1][1]
                cv2.line(output_image_frame,(x1,y1),(x2,y2),color,2)
        
        # Write the output video in the format mp4
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

        videoWriter.write(output_image_frame)
        cv2.imshow('Visualizing tracking trajectories', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
