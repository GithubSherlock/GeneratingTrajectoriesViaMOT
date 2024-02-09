"""
In 'video_path' set the video file to be processed, in 'output_path' set the output folder of the output frame images,
and in 'frame_interval' set the splitting of frame interval between each frame images. The frame interval is currently
set to 10 fps. If this program runs successfully a positive result will be displayed on the console.
Applied libs: cv2, os
Author: Shiqi Jiang, 15.06.2023
"""

import cv2
import os

# Compute frame number and fps of input video
def measure_video(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = 0  # Initialize frame counter
    all_frames = []

    while True:
        ret, frame = video.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count += 1

    print("Count the number of frames:", frame_count)
    print("There are", len(all_frames), "frames")  # Total number of frames

    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    if int(major_ver) >= 3:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    video.release()

# Bild a function to save video files as PNG images with a frame interval
def save_frames(video_path, output_path, frame_interval):
    os.makedirs(output_path, exist_ok=True) #  Check if the output folder exists, and create it if it does not.
    video_capture = cv2.VideoCapture(video_path)  # Open video file
    frame_count = 0  # Initializing the frame counter

    while True:
        ret, frame = video_capture.read()  # Read the frames of the video
        if not ret:  # Determine if the frame was successfully read
            break
        frame_count += 1
        if frame_count % frame_interval == 0:  # Save only frames of the specified interval
            filename = os.path.join(output_path, f"{frame_count:06d}.png")  # Build the file name to save the image
            cv2.imwrite(filename, frame)  # Save frames as PNG images
    video_capture.release()  # Release video capture object
    return True

# The main function
if __name__ == '__main__':
    #  Set frame interval, input video and output folder path
    video_path = './input/sportscheck_street_1.mp4'  # Set the video file path as input
    output_path = './output/output_sportscheck_street'  # Set the output folder path
    frame_interval = 10  # Set frame interval

    #  Run main functions
    measure_video(video_path)
    save_frames(video_path, output_path, frame_interval)  # Use the function to save the frame as PNG images
    if save_frames(video_path, output_path, frame_interval):  # If image frames saved successfully print a note
        print("Save successfully!")
