import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def find_path_from_floor(floor_mask):
    """
    Finds the path (centerline) based on the left and right boundaries of the floor mask.
    
    Args:
        floor_mask (numpy.ndarray): A binary mask where the floor is marked as 1 (or True).
    
    Returns:
        path (list): A list of (x, y) midpoints representing the centerline of the path in the detected floor area.
    """
    height, width = floor_mask.shape
    midpoints = []

    # Loop through each row of the mask
    for y in range(height):
        row = floor_mask[y, :]

        if np.any(row): # If there are non-zero pixels in the row
            # Find the leftmost and rightmost points in the row
            x_left = np.argmax(row)  # The first '1' from the left
            x_right = width - np.argmax(np.flip(row)) - 1  # The first '1' from the right

            if x_right > x_left:  # Only consider rows with valid boundaries
                # Compute the midpoint between left and right boundaries
                x_mid = (x_left + x_right) // 2
                midpoints.append((x_mid, y))

    return midpoints

def path_smoothing(path, smooth_factor=2):
    """
    Smooth the path using spline interpolation.
    
    Args:
        path (list): A list of (x, y) midpoints representing the path.
        smooth_factor (float): Smoothing factor for the spline. Higher values result in smoother curves.
        
    Returns:
        smoothed_path (list): A smoothed list of (x, y) midpoints.
    """
    if len(path) < 2:
        return path  # Not enough points to smooth

    # Extract x and y coordinates
    x_coords, y_coords = zip(*path)
    
    # Generate spline
    tck, _ = splprep([x_coords, y_coords], s=smooth_factor)
    
    # Generate new points on the spline
    new_points = np.linspace(0, 1, len(path))
    x_smooth, y_smooth = splev(new_points, tck)
    
    # Combine x and y back into a list of midpoints
    smoothed_path = list(zip(map(int, x_smooth), map(int, y_smooth)))

    # Find the point with the lowest y value
    lowest_point = min(smoothed_path, key=lambda point: point[1])
    # Extract the x value corresponding to the lowest y value
    path_position_x = lowest_point[0]
    
    return smoothed_path, path_position_x

def draw_path_on_image(image, path):
    """
    Draws the computed path on the image.
    
    Args:
        image (numpy.ndarray): The original image to draw the path on.
        path (list): The path as a list of (x, y) points.
    
    Returns:
        image_with_path (numpy.ndarray): The image with the path drawn on it.
    """
    for (x, y) in path:
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Draw a small circle at each midpoint
    return image
