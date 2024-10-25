import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def find_path_from_floor(floor_mask, threshold=10):
    """
    Finds the path (centerline) based on the left and right boundaries of the floor mask,
    ensuring that no midpoint is closer than a specific threshold to any boundary points.
    If the threshold is too small, the midpoint is adjusted accordingly.
    
    Args:
        floor_mask (numpy.ndarray): A binary mask where the floor is marked as 1 (or True).
        threshold (int): The minimum distance allowed between the midpoint and all annotations.
    
    Returns:
        path (list): A list of (x, y) midpoints representing the centerline of the path in the detected floor area.
    """
    height, width = floor_mask.shape
    midpoints = []

    # Loop through each row of the mask
    for y in range(height):
        row = floor_mask[y, :]

        if np.any(row):  # If there are non-zero pixels in the row
            # Get all the non-zero (annotated) points in the row
            annotated_points = np.where(row == 1)[0]

            if len(annotated_points) > 1:  # Only consider rows with more than one annotation
                x_left = annotated_points[0]  # The leftmost annotated point
                x_right = annotated_points[-1]  # The rightmost annotated point
                
                # Compute the initial midpoint between left and right boundaries
                x_mid = (x_left + x_right) // 2

                # Calculate the distance from the midpoint to all annotated points
                distances_to_annotations = np.abs(annotated_points - x_mid)

                # If the minimum distance to any annotation is less than the threshold, adjust the midpoint
                if np.min(distances_to_annotations) < threshold:
                    # Find the closest annotation point
                    closest_annotation_idx = np.argmin(distances_to_annotations)
                    closest_annotation = annotated_points[closest_annotation_idx]

                    # Adjust the midpoint so that it's at least `threshold` away from the closest annotation
                    if x_mid < closest_annotation:
                        x_mid = closest_annotation - threshold
                    else:
                        x_mid = closest_annotation + threshold

                    # Make sure x_mid stays within the boundaries of the row
                    x_mid = max(x_left + threshold, min(x_right - threshold, x_mid))

                # Append the valid or adjusted midpoint
                midpoints.append((x_mid, y))

    return midpoints

def path_smoothing(path, smooth_factor=2, number_of_control_points=10):
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

    # Find the point with the number_of_control_points lowest y values
    sorted_by_y = sorted(smoothed_path, key=lambda point: point[1])
    lowest_points = sorted_by_y[:number_of_control_points]
    average_x = sum(point[0] for point in lowest_points) / len(lowest_points)
    
    return smoothed_path, average_x

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
