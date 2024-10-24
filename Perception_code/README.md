# Robotic Perception

This project implements a perception system for a robot to navigate through corridors using floor detection, path planning, and control. It uses instance segmentation to detect drivable floor areas and a PID controller to adjust the robot's movement based on the detected path.

## Floor Detection Model

- **Model**: [`walt-floor_detection`](https://universe.roboflow.com/walt-snt93/walt-floor_detection) on Roboflow 
- **Model Type**: Roboflow 3.0 Instance Segmentation (Fast)
- **mAP**: 99.5%
- **Trained on**: 3k+ labeled images

## Key Components

1. **Floor Detection**: The Roboflow model performs instance segmentation on the floor.
2. **Path Finding**: Midpoints of the detected floor boundaries are used to create a drivable path and smoothed after.
3. **PID Controller**: The controller calculates a turning rate based on the robot's position relative to the path.

## Requirements

To install the necessary dependencies, run:

```bash
pip install opencv-python numpy roboflow supervision scipy inference
