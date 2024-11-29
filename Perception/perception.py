import cv2
import os
import numpy as np
import supervision as sv
import inference
from modules.fps import calculate_fps, add_fps_on_image
from modules.occupancy_grid import OccupancyGrid
from modules.fetch_imu_data import fetch_imu_data


def perception(video_source=0, output_file='output.mp4', version=1, confidence=50, overlap=25, imu_ip="", imu_port=""):
    """Continue until 'e' is pressed."""

    # Initialize the model from Roboflow
    detect_floor = inference.get_model(f"walt-floor-and-stair-detection/{version}")
    detect_floor.confidence = confidence
    detect_floor.overlap = overlap

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError("Failed to open video source.")

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, 10.0, (frame_width, frame_height))

    # Initialize occupancy grid
    occupancy_grid = OccupancyGrid(grid_size=1000, resolution=5)

    # To calculate FPS
    prev_time = 0

    # Initialize robot position and velocity
    robot_x, robot_y = 500, 999  # Bottom-middle of the grid
    robot_theta = 0
    velocity_x, velocity_y = 0, 0  # Initial velocities

    while True:
        # Fetch IMU data
        imu_data = fetch_imu_data(imu_ip, imu_port)

        # Extract acceleration and angular velocity
        #accel_x, accel_y = imu_data["accel"][:2]
        #gyro_z = imu_data["gyro"][2]
        accel_x, accel_y = 0, 0
        gyro_z = 0

        # Calculate the time step (fps as frame interval)
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        dt = 1 / 10  # Assuming 10 FPS, or calculate dynamically if desired
        prev_time = current_time

        # Update robot pose using IMU data
        robot_theta = occupancy_grid.update_orientation(robot_theta, gyro_z, dt)
        robot_x, robot_y, velocity_x, velocity_y = occupancy_grid.update_position(
            robot_x, robot_y, accel_x, accel_y, velocity_x, velocity_y, dt
        )

        # Capture the latest image
        ret, image = cap.read()
        if not ret:
            print("Failed to grab image")
            break

        # Perform floor detection on the captured image
        pred_floor = detect_floor.infer(image)[0]

        # Extract the floor mask
        detections = sv.Detections.from_inference(pred_floor)
        floor_mask = detections.mask[0].astype(np.uint8) * 255  # Assuming floor is at index 0
        
        # Update occupancy grid with the floor mask
        occupancy_grid.update_grid(floor_mask)

        # Visualize occupancy grid
        occupancy_grid.visualize_grid()

        # Annotate and display the original image
        polygon_annotator = sv.PolygonAnnotator()
        annotated_image = polygon_annotator.annotate(scene=image, detections=detections)

        # Calculate FPS
        fps, prev_time = calculate_fps(prev_time)
        # Put the FPS on the image
        add_fps_on_image(annotated_image, fps)

        # Write the image into the output file
        out.write(annotated_image)

        # Display the image
        cv2.imshow('Floor Detection', annotated_image)

        # Press 'e' to exit the loop and stop the video stream
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Release the video capture object and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

output_file = r"C:\Users\leon-\Videos\WALT\predicted_output.mp4"
video_source=0
version = 1
imu_ip = "192.168.86.20"
imu_port = 8080

perception(video_source, output_file, version, imu_ip, imu_port)
