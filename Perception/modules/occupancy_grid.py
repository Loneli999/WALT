import numpy as np
import matplotlib.pyplot as plt
import cv2

class OccupancyGrid:
    def __init__(self, grid_size=1000, resolution=5):
        """
        Initialize an occupancy grid.
        
        :param grid_size: Size of the grid (e.g., 1000x1000 cells)
        :param resolution: Size of each cell in cm (e.g., 5 cm per cell)
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.full((grid_size, grid_size), -1, dtype=np.int8)  # -1 for unknown

        # Assume robot starts at the center
        self.robot_x = grid_size // 2
        self.robot_y = grid_size // 2
        self.robot_theta = 0  # Robot orientation in radians
    
    def update_orientation(self, robot_theta, gyro_z, dt):
        """
        Update robot orientation using gyroscope data.

        :param robot_theta: Current orientation of the robot in radians
        :param gyro_z: Angular velocity around the z-axis in rad/s
        :param dt: Time step in seconds
        :return: Updated orientation in radians
        """
        robot_theta += gyro_z * dt  # Integrate angular velocity over time
        return robot_theta

    def update_position(self, robot_x, robot_y, accel_x, accel_y, velocity_x, velocity_y, dt):
        """
        Update robot position using acceleration data.

        :param robot_x: Current x-position of the robot in grid coordinates
        :param robot_y: Current y-position of the robot in grid coordinates
        :param accel_x: Acceleration along the x-axis in m/s²
        :param accel_y: Acceleration along the y-axis in m/s²
        :param velocity_x: Current velocity along the x-axis in m/s
        :param velocity_y: Current velocity along the y-axis in m/s
        :param dt: Time step in seconds
        :return: Updated x, y positions in grid coordinates and updated velocities
        """
        # Convert acceleration to velocity
        velocity_x += accel_x * dt
        velocity_y += accel_y * dt

        # Convert velocity to position
        robot_x += int(velocity_x * dt / self.resolution)
        robot_y -= int(velocity_y * dt / self.resolution)  # Negative because grid y decreases upwards

        return robot_x, robot_y, velocity_x, velocity_y

    def update_grid(self, floor_mask):
        """
        Update the occupancy grid using the detected floor mask.
        
        :param floor_mask: Binary mask of the detected floor (same size as the camera frame)
        """
        mask_height, mask_width = floor_mask.shape

        for y in range(mask_height):
            for x in range(mask_width):
                if floor_mask[y, x] > 0:  # Floor detected
                    # Convert mask coordinates to grid coordinates
                    global_x = self.robot_x + (x - mask_width // 2)
                    global_y = self.robot_y - y  # Project forward in front of the robot
                    
                    if 0 <= global_x < self.grid_size and 0 <= global_y < self.grid_size:
                        self.grid[global_y, global_x] = 0  # Mark as free space


    def visualize_grid(self, window_name='Occupancy Grid'):
        """
        Visualize the occupancy grid.
        """
        display_grid = (self.grid + 1) * 127  # Scale values for display (-1 -> 0, 0 -> 127)
        display_grid = display_grid.astype(np.uint8)
        cv2.imshow(window_name, display_grid)
