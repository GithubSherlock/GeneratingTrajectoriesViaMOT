# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:44:29 2024

@author: li
"""

import numpy as np
import matplotlib.pyplot as plt

def moving_average_trajectory(points, window_size):
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
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Original trajectory
trajectory = np.array([(1, 2), (2, 4), (3, 5), (4, 6), (5, 8)])

# Smoothed trajectory
smoothed_trajectory = np.array(moving_average_trajectory(trajectory, window_size=3))

# Plotting
plt.figure(figsize=(8, 6))

# Plot original trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label='Original Trajectory')

# Plot smoothed trajectory
plt.plot(smoothed_trajectory[:, 0], smoothed_trajectory[:, 1], 'o-', label='Smoothed Trajectory')

# Mark start and end points
plt.scatter(*trajectory[0], color='green', label='Start')
plt.scatter(*trajectory[-1], color='red', label='End')

# Set labels and title
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Original and Smoothed Trajectories')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

