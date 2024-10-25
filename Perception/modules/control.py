import cv2
import numpy as np
import time

class PIDController:
    def __init__(self, K_p, K_i, K_d):
        self.K_p = K_p  # Proportional gain
        self.K_i = K_i  # Integral gain
        self.K_d = K_d  # Derivative gain
        
        self.previous_error = 0  # To store the previous error for derivative calculation
        self.integral = 0        # To accumulate the integral of the error
        
    def calculate_turning_rate(self, robot_position, path_position_x):
        """
        Calculate the turning rate using a PID controller.
        
        Args:
            robot_position (int): The x-coordinate of the robot's current position.
            path_position (int): The x-coordinate of the target path.
        
        Returns:
            turning_rate (int): The computed turning rate (-100 to 100).
        """
        # Calculate the current error
        error = path_position_x - robot_position
        
        # Update the integral (sum of errors)
        self.integral += error
        
        # Calculate the derivative (change in error)
        derivative = error - self.previous_error
        
        # Calculate the turning rate using PID control
        turning_rate = (self.K_p * error) + (self.K_i * self.integral) + (self.K_d * derivative)
        
        # Clamp the turning rate to the range [-100, 100]
        turning_rate = max(-100, min(100, turning_rate))
        
        # Update the previous error
        self.previous_error = error
        
        return int(turning_rate)
    
def send_turning_rate_to_robot(turning_rate):
    """
    Send the computed turning rate to the robot.

    Args:
        turning_rate (int): The computed turning rate (-100 to 100).
    """
    print(f"Sending turning rate: {turning_rate}")
    # Add here communication code...