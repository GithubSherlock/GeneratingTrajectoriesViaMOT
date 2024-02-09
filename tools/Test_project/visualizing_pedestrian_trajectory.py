"""
The data of each frame is collated into a list by reading XML files in the folder path "./annotation",
which generated via labelling software “LabelImg”. The image coordinates of a pedestrian are then built from the data
in the list, followed by the projective transformation to build a map in satellite view and the geographic coordinates
of the pedestrian in the map. Finally, two visualisations were implemented and a choice of whether to output a frame
animation or a gif animation was made by entering "1" or "2" in the console. The gif animation is output in the folder
path " . /output".
Applied libs: cv2, os, numpy, matplotlib, xml.etree.ElementTree
Author: Shiqi Jiang, 17.06.2023
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.animation import FuncAnimation

# Parsing the given XML files
def parse_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []
    for child in root:
        if child.tag == 'filename':
            data.append(child.text)
        elif child.tag == 'path':
            data.append(child.text)
        elif child.tag == 'object':
            for obj_child in child:
                if obj_child.tag == 'name':
                    data.append(obj_child.text)
                elif obj_child.tag == 'bndbox':
                    bbox_data = []
                    for bbox in obj_child:
                        bbox_data.append(int(bbox.text))
                    data.extend(bbox_data)

    return data

# Processing XML files in a given folder
def process_xml_files(folder_path):
    file_names = [f'frame_{i}' for i in range(10, 1411)]  # maybe change range
    frame_data = {}
    for file_name in file_names:
        xml_file = os.path.join(folder_path, f'{file_name}.xml')
        if os.path.exists(xml_file):
            data = parse_xml_file(xml_file)
            frame_data[file_name] = data

    return frame_data

# Compute image coordinates of pedestrians
def build_image_coordinates(frame_data):
    Location_frame = []
    Location_x = []
    Location_y = []
    for key in frame_data:
        frame_number = int(key.split('_')[1])  # Extract the numeric part of the file name
        Location_frame.append(f'Location_{frame_number}')
        Location_x.append((frame_data[key][3] + frame_data[key][5]) / 2)
        Location_y.append(frame_data[key][6])
    Locations = [Location_frame, [Location_x, Location_y]]

    return Locations

# Perform perspective transformations
def perform_perspective_transform(image_path, src, dest, width, height, Locations):
    image = cv2.imread(image_path, 1)
    mtx = cv2.getPerspectiveTransform(np.float32(src), np.float32(dest))
    output_image = cv2.warpPerspective(image, mtx, (width, height))
    # Build pedestrian's image coordinates
    person_IC = []
    for i in range(len(Locations[0])):
        x = Locations[1][0][i]
        y = Locations[1][1][i]
        person_IC.append((x, y))
    original = np.float32([person_IC])
    converted = cv2.perspectiveTransform(original, mtx)
    # Organize the coordinates of the transformed person as a list and as an integer
    Location_x = []
    Location_y = []
    Location_frame_ic = []
    for i in range(len(converted[0])):
        frame_number = Locations[0][i]
        x = converted[0][i][0]  # Extract the x coordinate of the i-th location
        y = converted[0][i][1]  # Extract the y coordinate of the i-th location
        Location_frame_ic.append(frame_number)
        Location_x.append(x)  # Add the x coordinate to the Location_x list
        Location_y.append(y)  # Add the y coordinate to the Location_y list
    Location_x_ic = [int(x) for x in Location_x]  # Convert to integer
    Location_y_ic = [int(y) for y in Location_y]  # Convert to integer
    # Combine Location_frame_ic, Location_x and Location_y to create locations list
    Locations_ic = list(zip(Location_frame_ic, Location_x_ic, Location_y_ic))

    return output_image, Locations_ic

# Save the transformed image
def save_transformed_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f"Transformed image saved at: {output_path}")

# Display the original image and the transformed image
def display_images(original_image, transformed_image):
    cv2.imshow('Org map', original_image)
    cv2.imshow('Perspective transformed map', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Perform visualization with a frame animation
def visualize_animation(map_image, locations):
    # Set marker properties
    marker_color = (0, 0, 255)  # Red color
    marker_radius = 5
    marker_thickness = -1  # Filled circle
    # Set text properties
    text = "person"
    text_color = (0, 0, 255)  # Red color
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.5
    text_thickness = 1
    # Set line properties
    line_color = (0, 255, 0)  # Green color
    line_thickness = 2
    # Create output folder
    output_folder = "./output/animation_frames"
    os.makedirs(output_folder, exist_ok=True)
    # Create empty trajectory image
    trajectory_image = map_image.copy()
    # Decrease brightness of the input image
    brightness_factor = 0.1  # Adjust this value to control the brightness
    trajectory_image = cv2.convertScaleAbs(trajectory_image, alpha=brightness_factor, beta=0)
    # Create frame animation
    for frame_idx, (location_name, x, y) in enumerate(locations):
        frame_image = map_image.copy()
        # Draw trajectory
        if frame_idx > 0:
            prev_x, prev_y = locations[frame_idx - 1][1], locations[frame_idx - 1][2]
            cv2.line(trajectory_image, (prev_x, prev_y), (x, y), line_color, line_thickness)
        # Combine frame image and trajectory image
        frame_with_trajectory = cv2.addWeighted(frame_image, 1, trajectory_image, 1, 0)
        # Draw marker
        cv2.circle(frame_with_trajectory, (x, y), marker_radius, marker_color, marker_thickness)
        # Draw text
        text_position = (x + 10, y - 10)  # Next to the marker
        cv2.putText(frame_with_trajectory, text, text_position, text_font, text_scale, text_color, text_thickness,
                    cv2.LINE_AA)
        # Save frame image
        actual_frame_path = os.path.join(output_folder, f"{output_folder}/frame_{frame_idx}.png")
        cv2.imwrite(actual_frame_path, frame_with_trajectory)
        # Display frame
        cv2.imshow("Visualizing pedestrian trajectory", frame_with_trajectory)
        # Wait for key press
        key = cv2.waitKey(1000)  # 1 second delay
        # Check if Esc or Space key is pressed
        if key == 27 or key == 32:
            break
    # Close all windows
    cv2.destroyAllWindows()
    print(f"Frames saved in {output_folder}")

# Perform visualization with a gif animation file as output
def visualize_animation_in_gif(map_image, locations):
    # Set marker style
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 255)
    thickness = 1
    # Create output folder
    output_folder = "./output/output_animation_in_gif"
    os.makedirs(output_folder, exist_ok=True)
    # Sort locations based on frame index
    sorted_locations = sorted(locations, key=lambda loc: int(loc[0].split('_')[1]))
    # Create a list to store the intermediate positions
    intermediate_positions = []
    for frame_index, x, y in sorted_locations:
        # Draw marker and text on the map image
        annotated_image = map_image.copy()
        cv2.circle(annotated_image, (x, y), 5, color, -1)
        cv2.putText(annotated_image, "person", (x + 10, y - 10), font, font_scale, color, thickness)
        # Save current frame image
        output_path = os.path.join(output_folder, f"frame_{frame_index}.png")
        cv2.imwrite(output_path, annotated_image)
        # Add the current position to intermediate positions
        intermediate_positions.append((x, y))
    # Plot the trajectory on the map image using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(map_image)
    line, = ax.plot([], [], 'g-')
    point, = ax.plot([], [], 'ro')
    annotation = ax.text(1, -1, 'Person', color='red', backgroundcolor='none')
    ax.set_xlim(0, map_image.shape[1])
    ax.set_ylim(map_image.shape[0], 0)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Visualizing pedestrian trajectory')

    def init():
        line.set_data([], [])
        point.set_data([], [])
        annotation.set_text('')
        return line, point, annotation

    def animate(frame):
        x = [pos[0] for pos in intermediate_positions[:frame]]
        y = [pos[1] for pos in intermediate_positions[:frame]]
        line.set_data(x, y)
        point.set_data(intermediate_positions[frame][0], intermediate_positions[frame][1])
        annotation.set_position((intermediate_positions[frame][0] + 10, intermediate_positions[frame][1] - 10))
        annotation.set_text('Person')
        return line, point, annotation

    animation = FuncAnimation(fig, animate, frames=len(sorted_locations), init_func=init, interval=1000, blit=True)
    # Save the animation as a GIF
    animation_path = './output/animation.gif'
    animation.save(animation_path, writer='pillow', fps=1)
    print(f"Animation saved at: {animation_path}")
    # Display the animation
    plt.show()

# The main function
def main():
    # Set the folder path
    input_folder_path = './annotation'  # Set annotation folder path for labeled XML files

    # Process XML files and get frame data
    frame_data = process_xml_files(input_folder_path)

    # Create image coordinate data
    Location = build_image_coordinates(frame_data)

    # Read input images and set the width and height of the output image
    image_path = './output/output_frames/frame_1450.png'  # Choose a correct map image to transform
    image_org = cv2.imread(image_path, 1)
    width = 1920
    height = 1080

    # Input the pixel coordinates of the reference point in the image
    src = [[458, 48], [1418, 32], [1588, 211], [542, 184]]
    dest = np.float32([[212, 214], [633, 236], [527, 474], [266, 405]])

    # Perform perspective transformations
    perspective_transformed_map = perform_perspective_transform(image_path, src, dest, width, height, Location)[0]
    Locations_ic = perform_perspective_transform(image_path, src, dest, width, height, Location)[1]

    # Specify the save path and file name and save the transformed image
    output_path = './output/perspective_transformed_map.png'
    save_transformed_image(perspective_transformed_map, output_path)

    """ Show the map after the projective transformation as well as the original image.
    # Display the original image and the transformed image
    display_images(image_org, perspective_transformed_map)
    """

    # A choice for visualization with output only frame animation or gif animation
    choice = input("Is the visualization you want to implement only frame animation or gif animation?\n"
                   "For the former, enter '1' in the console, for the latter, enter '2'. ")

    if choice == '1':
        print("If you want, please press Esc or Space to exit the animation after the animation starts.\n"
              "Here we go!")
        cv2.waitKey(1000)
        print("3")
        cv2.waitKey(1000)
        print("2")
        cv2.waitKey(1000)
        print("1")
        cv2.waitKey(1000)
        visualize_animation(perspective_transformed_map, Locations_ic)
        exit(0)
    elif choice == '2':
        visualize_animation_in_gif(perspective_transformed_map, Locations_ic)
        print("Please check the gif file in folder ./output")
        exit(0)
    else:
        exit(0)

    """ The following code functions as a separate implementation of visualization
    # Reset the image and perform visualization with a frame animation
    visualize_animation(perspective_transformed_map, Locations_ic)

    # Perform visualization with a gif animation file as output
    visualize_animation_in_gif(perspective_transformed_map, Locations_ic)
    """

# Run the main function
if __name__ == "__main__":
    main()
