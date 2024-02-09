# -*- coding: utf-8 -*-
"""
This program is designed to animate object trajectories in a video through projective transformation. Users input pixel coordinates 
of a reference point, a dataset folder path, and an output video path. The program processes the dataset, applying the projective 
transformation to track each object's movement. Key features include a control board setting reference points, window size for moving 
averages, and plotting colours. This comprehensive program serves as a tool for detailed and smoothed visualization of object movements 
in video data, leveraging projective transformation for accurate tracking.
@author: Shiqi Jiang
Created on Wed Jan 25 08:32:34 2024
"""

import os
import numpy as np
import cv2

# Constant Control Board
BACKGROUND_PATH = './background/googlemap_sportscheck.png' # perspectiveMap_mensa
DATASET_PATH = './datasets/continuousFrame/sportscheck_353/iou0.5_conf0.6/' # _with_10interval
RESULT_PATH = './output/sportscheck_projection.mp4'

# Input the pixel coordinates of the reference point in the image, for mensa
# SCR = np.array([[396, 525], [688, 389], [1114, 350], [1388, 947]], dtype=np.float64)
# DEST = np.array([[353, 388], [665, 385], [960, 705], [180, 777]], dtype=np.float64)
# For sportscheck
SCR = np.array([[901, 370], [1553, 362], [1776, 843], [259, 500]], dtype=np.float32)
DEST = np.array([[820, 788], [327, 833], [207, 286], [1066, 370]], dtype=np.float32)

WINDOW_SIZE = 5
# Constant Control Board

def get_img_size(img_path):
    """
    Function to get the size of an image using OpenCV.

    Parameters
    ----------
    img_path : str
        Path to the image file.

    Returns
    -------
    tuple
        A tuple containing the height and width of the image.

    Raises
    ------
    FileNotFoundError
        If the file at the specified path does not exist.
    ValueError
        If the image cannot be opened or does not have the correct dimensions.
    """
    googleMap = cv2.imread(img_path)
    height, width, _ = googleMap.shape
    return width, height

def read_dataset(DATASET_PATH):
    """
    Reads the dataset from the given directory.

    Parameters
    ----------
    DATASET_PATH : str
        The path to the dataset directory.

    Returns
    -------
    list
        A list of tuples containing the data from each line in the dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset directory does not exist or contains no files.
    """
    txt_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.txt')]
    if txt_files:
        file_path = os.path.join(DATASET_PATH, txt_files[0])
        with open(file_path, 'r') as file:
            dataset = [line.strip().split(',') for line in file]
            return dataset
    else:
        return print("No txt files found in the directory.")

class Coordinate:
    def __init__(self, x, y):
        """
        Initialize a new Coordinate object.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        """
        self.x = x
        self.y = y

    def centre_pos(self, x1, y1, w, h, cls_id, obj_id):
        """
        Calculate the center position of a bounding box.

        Args:
            x1 (float): The x-coordinate of the top-left corner of the bounding box.
            y1 (float): The y-coordinate of the top-left corner of the bounding box.
            w (float): The width of the bounding box.
            h (float): The height of the bounding box.
            cls_id (int): The class ID of the object.
            obj_id (int): The object ID of the object.

        Returns:
            tuple: A tuple containing the x-coordinate and y-coordinate of the center of the bounding box.

        """
        cx = int(x1 + w/2)
        cy = int(y1 + h) # (y2-y1)/2+y1 for cy of bbox

        return cx, cy, cls_id, obj_id

    def coordinates_inSatelliteMap(self, src, dest, coordinates):
        """
        Transform coordinates from one coordinate system to another using perspective transformation.

        Args:
            src (list): A list of four source coordinates, represented as [x1, y1, x2, y2].
            dest (list): A list of four destination coordinates, represented as [x1, y1, x2, y2].
            coordinates (list): A list of two coordinates to be transformed, represented as [x, y].

        Returns:
            list: A list containing the transformed coordinates.

        """
        MTX = cv2.getPerspectiveTransform(np.float32(src), np.float32(dest))
        coordinates_np = np.array([coordinates], dtype=np.float32).reshape(-1, 1, 2)
        transformed_coordinates = cv2.perspectiveTransform(coordinates_np, MTX)

        return transformed_coordinates

def smoothing_filter(dataset):
    """
    This function applies a moving average to the trajectory of objects in a dataset.

    Parameters:
    dataset (list): A list of tuples containing the data from each line in the dataset. Each tuple contains the frame number, x-coordinate, y-coordinate, class ID, and object ID.

    Returns:
    list: A list of tuples containing the smoothed data. Each tuple contains the frame number, x-coordinate, y-coordinate, class ID, and object ID.

    """
    # Create a dictionary to store the tracks for each ID object
    trajectories_dict = {}
    # Iterate over the transformed data
    for data in dataset:
        frame_number, x, y, cls_id, obj_id = data
        # Use (cls_id, obj_id) as key
        key = (cls_id, obj_id)
        # If the key does not exist in the dictionary, add the key and initialise to an empty list
        if key not in trajectories_dict:
            trajectories_dict[key] = []
        # Add track points to the corresponding list in trajectories_dict
        trajectories_dict[key].append((frame_number, (x, y)))
    # Create an empty list to store smoothed trajectories
    smoothed_data_np = []

    # Iterate over each track in the dictionary
    for key, trajectory_points in trajectories_dict.items():
        # Extract the original frame list
        frame_numbers = [point[0] for point in trajectory_points]
        # Extract coordinate points (remove frames)
        trajectory_coords = [point[1] for point in trajectory_points]
        # If the track length is smaller than the window size, the original track is used directly
        if len(trajectory_coords) < WINDOW_SIZE:
            for point in trajectory_points:
                smoothed_data_np.append([point[0], point[1][0], point[1][1], key[0], key[1]])
            continue
        # Apply a moving average function to smooth trajectory
        smoothed_trajectory = moving_average_trajectory(trajectory_coords, WINDOW_SIZE)
        # Combine the smoothed trajectory points with the number of start frames and end frames
        smoothed_trajectory_full = [(frame_numbers[0], smoothed_trajectory[0])]
        smoothed_trajectory_full += zip(frame_numbers[1:-1], smoothed_trajectory[1:])
        smoothed_trajectory_full.append((frame_numbers[-1], smoothed_trajectory[-1]))
        # Adding smoothed tracks to the list
        cls_id, obj_id = key
        for frame_number, coords in smoothed_trajectory_full:
            smoothed_data_np.append([frame_number, coords[0], coords[1], cls_id, obj_id])
    # Converting list to numpy array
    smoothed_data_np = np.array(smoothed_data_np)
    # Define a mapping function that converts cls_id to sort weights
    cls_id_sorting_key = {'person': 1, 'car': 2, 'truck': 3, 'bicycle': 4}
    # Create a new array where cls_id is replaced by its sort weights
    cls_id_as_numbers = np.array([cls_id_sorting_key[cls_id] for cls_id in smoothed_data_np[:, 3]])
    # Sorting with lexsort
    # First sorted by obj_id in ascending order, then by cls_id type, and finally by number of frames
    sort_indices = np.lexsort((smoothed_data_np[:, -1].astype(int), cls_id_as_numbers, smoothed_data_np[:, 0].astype(int)))
    # Applying a Sorted Index
    sorted_smoothed_data_np = smoothed_data_np[sort_indices]

    return sorted_smoothed_data_np
    
def moving_average_trajectory(points, window_size):
    """
    This function takes a list of points (x,y coordinates) and a window size as input, and returns a smoothed version of the trajectory by applying a moving average to the x and y coordinates.

    Parameters:
    points (list): A list of points (x,y coordinates)
    window_size (int): The size of the window for the moving average

    Returns:
    list: A smoothed version of the trajectory, where each point is a tuple of (x,y) coordinates

    Raises:
    ValueError: If the window size is greater than the length of the trajectory
    """
    if len(points) < window_size:
        raise ValueError("Window size should be smaller than the length of the trajectory.")

    # Extract x and y coordinates
    x_coords, y_coords = zip(*points)

    # Apply moving average separately to x and y coordinates
    smoothed_x = moving_average(x_coords, window_size)
    smoothed_y = moving_average(y_coords, window_size)

    # Combine the smoothed x and y coordinates into a new trajectory
    smoothed_trajectory = list(zip(smoothed_x, smoothed_y))

    # Add the first and last points of the original trajectory to the smoothed trajectory
    smoothed_trajectory = [(x_coords[0], y_coords[0])] + smoothed_trajectory + [(x_coords[-1], y_coords[-1])]

    return smoothed_trajectory

def moving_average(data, window_size):
    """
    This function takes a numpy array and a window size as input, and returns a smoothed version of the data by applying a moving average to the data points.

    Parameters:
    data (np.ndarray): The input data, which should be a 1-dimensional numpy array
    window_size (int): The size of the window for the moving average

    Returns:
    np.ndarray: A smoothed version of the input data, where each point is the average of the surrounding points

    Raises:
    ValueError: If the window size is greater than the length of the data
    """
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def visualising_animation(transformed_data_np, color_dict, WIDTH, HEIGHT, fps=24):
    """
    This function creates an animation of the trajectory of objects in a video using the projective transformation.

    Parameters:
    transformed_data_np (np.ndarray): A numpy array containing the transformed data, where each row represents a frame and the columns are x, y, 
    class ID, object ID, and frame number.
    color_dict (dict): A dictionary that maps class IDs to colors.
    WIDTH (int): The width of the output video.
    HEIGHT (int): The height of the output video.
    fps (int, optional): The frame rate of the output video. Defaults to 24.

    Returns:
    None

    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (WIDTH, HEIGHT))
    
    frame_numbers = transformed_data_np[:, 0].astype(int)
    max_frame_number = frame_numbers.max()

    trails_dict = {}

    # Create animation
    for frame_number in range(1, max_frame_number + 1):  # Starting from frame 1 to the end
        frame_data = transformed_data_np[frame_numbers == frame_number]
        background = cv2.imread(BACKGROUND_PATH)
        
        # Processing the object in the current frame
        current_trails_dict = {}
        for data in frame_data:
            _, x, y, cls_id, obj_id = data
            key = (cls_id, obj_id)  # Create a key for the current frame
            color = color_dict[cls_id]
            
            # Plotting object points and texts
            position = (int(float(x)), int(float(y)))
            cv2.circle(background, position, 5, color, -1)
            cv2.putText(background, f'{cls_id} {obj_id}', (position[0] + 10, position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Update or add track points
            if key not in trails_dict:
                trails_dict[key] = []
            trails_dict[key].append(position)
            
            # Add objects in the current frame to current_trails_dict
            current_trails_dict[key] = trails_dict[key]

        # Plot all trajectories
        for key, trail in trails_dict.items():
            for i in range(1, len(trail)):
                cv2.line(background, trail[i - 1], trail[i], color_dict[key[0]], 2)

        # Update the track dictionary
        trails_dict = current_trails_dict
        
        # Display the current frame
        cv2.imshow('Projective animation', background)
        
        # Write current frame to video file
        video_writer.write(background)
        
        # Press 'q' or 'Esc' to exit
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
    
    # 释放资源并关闭窗口
    video_writer.release()
    cv2.destroyAllWindows()

def visualising_animation_with_10interval(transformed_data_np, color_dict, WIDTH, HEIGHT, fps=3):
    """
    This function creates an animation of the trajectory of objects in a video using the projective transformation.

    Parameters:
    transformed_data_np (np.ndarray): A numpy array containing the transformed data, where each row represents a frame and the columns are x, y, 
    class ID, object ID, and frame number.
    color_dict (dict): A dictionary that maps class IDs to colors.
    WIDTH (int): The width of the output video.
    HEIGHT (int): The height of the output video.
    fps (int, optional): The frame rate of the output video. Defaults to 24.

    Returns:
    None

    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (WIDTH, HEIGHT))
    
    # Get the frame count column from transformed_data_np and get unique frames by interval
    unique_frame_numbers = np.unique(transformed_data_np[:, 0].astype(int))

    # Dictionary storing tracks, tuple with keys (cls_id, obj_id)
    trails_dict = {}

    # Create the animation
    for frame_number in unique_frame_numbers:  # Starting from frame 1 to the end
        frame_data = transformed_data_np[transformed_data_np[:, 0].astype(int) == frame_number]
        background = cv2.imread(BACKGROUND_PATH)
        
        # Processing the object in the current frame
        current_trails_dict = {}
        for data in frame_data:
            _, x, y, cls_id, obj_id = data
            key = (cls_id, obj_id)  # Create a key for the current frame
            color = color_dict[cls_id]
            
            # Plotting object points and texts
            position = (int(float(x)), int(float(y)))
            cv2.circle(background, position, 5, color, -1)
            cv2.putText(background, f'{cls_id} {obj_id}', (position[0] + 10, position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Update or add track points
            if key not in trails_dict:
                trails_dict[key] = []
            trails_dict[key].append(position)
            
            # Add objects in the current frame to current_trails_dict
            current_trails_dict[key] = trails_dict[key]

        # Plot all trajectories
        for key, trail in trails_dict.items():
            for i in range(1, len(trail)):
                cv2.line(background, trail[i - 1], trail[i], color_dict[key[0]], 2)

        # Update the track dictionary
        trails_dict = current_trails_dict
        
        # Display the current frame
        cv2.imshow('Projective animation with 10 interval', background)
        
        # Write current frame to video file
        video_writer.write(background)
        
        # Press 'q' or 'Esc' to exit
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
    
    # Release resources and close the window
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    WIDTH, HEIGHT = get_img_size(BACKGROUND_PATH)
    # Dictionary to store the trail points of each object (choose clolors that you want:https://tool.oschina.net/commons?type=3)
    color_dict = {'person': (255, 165, 0), 'car': (238, 221, 130), 'bus': (255, 0, 0),
                  'truck': (139, 117, 0), 'bicycle': (0, 0, 128), 'motorbike': (0, 191, 255)}
    dataset = read_dataset(DATASET_PATH)
    if not dataset:
        raise ValueError("No data found in the txt file or dataset is empty.")
        
    coord = Coordinate(0, 0)

    numerical_data = np.asarray([row[:6] for row in dataset], dtype=np.float32)
    cls_ids = np.asarray([row[-1] for row in dataset])

    pre_smoothing_data = []
    for data_row, cls_id in zip(numerical_data, cls_ids):
        frame_number, obj_id, x1, y1, w, h = data_row
        cx, cy, cls_id, obj_id = coord.centre_pos(x1, y1, w, h, cls_id, obj_id)
        pre_smoothing_data.append([int(frame_number), cx, cy, cls_id, int(obj_id)])
    pre_smoothing_data = np.asarray(pre_smoothing_data)
    
    # sorted_smoothed_data_np = pre_smoothing_data # without smoothing
    sorted_smoothed_data_np = smoothing_filter(pre_smoothing_data)

    transformed_data = []
    for line in sorted_smoothed_data_np:
        frame_number, _, _, cls_id, obj_id = line
        coordinates = [line[1], line[2]]
        transformed_coordinates = coord.coordinates_inSatelliteMap(SCR, DEST, coordinates)
        transformed_data.append([int(frame_number), np.float32(transformed_coordinates[0][0][0]), np.float32(transformed_coordinates[0][0][1]), cls_id, int(obj_id)])
    transformed_data_np = np.asarray(transformed_data)

    visualising_animation(transformed_data_np, color_dict, WIDTH, HEIGHT)
    #visualising_animation_with_10interval(transformed_data_np, color_dict, WIDTH, HEIGHT)