# Perception

This project implements a perception system for a robot to navigate through corridors using floor detection, path planning, and high-level control. It uses instance segmentation to detect drivable floor areas and a PID controller to adjust the robot's movement based on the detected path.

## Floor Detection Model

- **Model**: [`walt-floor_detection`](https://universe.roboflow.com/walt-snt93/walt-floor-and-stair-detection/model/4) on Roboflow 
- **Model Type**: Yolov11
- **mAP**: 97.6%
- **Trained on**: 5k+ labeled images

## Key Components

1. **Floor and Stairs Detection**: The Yolo model performs instance segmentation on the floor and stairs.
2. **Path Finding**: Midpoints of the detected floor boundaries are used to create a drivable path and smoothed after.
3. **PID Controller**: The controller calculates a turning rate based on the robot's position relative to the path.
