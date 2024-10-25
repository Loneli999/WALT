import cv2
import os
import numpy as np
import supervision as sv
from inference import get_model
from modules.FPS import calculate_fps, add_fps_on_image
from modules.path_finder import find_path_from_floor, path_smoothing, draw_path_on_image
from modules.control import PIDController, send_turning_rate_to_robot


def perception(video_source=0, output_file='output.mp4', confidence = 50, overlap = 25, smooth_factor=2, number_of_control_points = 10, K_p=0.5, K_i=0.1, K_d=0.05):
    """Continue until 'e' is pressed."""

    # Initialize the floor detection model from Roboflow
    api_key = os.getenv("ROBOFLOW_WALT_FLOOR_DETECTION_API_KEY")
    detect_floor = get_model(f"walt-floor_detection/{version}", api_key=api_key)
    detect_floor.confidence = confidence
    detect_floor.overlap = overlap
    # Initialize PID-Controller
    pid_controller = PIDController(K_p, K_i, K_d)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError("Failed to open video source.")

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, 10.0, (frame_width, frame_height))
    
    # To calculate FPS
    prev_time = 0

    while True:
        # Capture the latest image
        ret, image = cap.read()
        if not ret:
            print("Failed to grab image")
            break

        # Perform floor detection on the captured image
        pred_floor = detect_floor.infer(image)[0]

        # Create an masked image
        detections = sv.Detections.from_inference(pred_floor)
        polygon_annotator = sv.PolygonAnnotator()
        annotated_image = polygon_annotator.annotate(scene=image, detections=detections)

        # Get the segmentation mask from the detections, compute the path and the turning_rate
        if detections.mask is not None:
            # Merge all masks into a single mask
            floor_mask = np.any(detections.mask, axis=0).astype(np.uint8) * 255  # Combine masks
            # Convert to binary (if needed)
            _, floor_mask = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)
            # Find the path from the detected floor
            path = find_path_from_floor(floor_mask)
            # Apply smoothing to the path and getting the x-position of the path that is the lowest in the image
            path, path_position_x = path_smoothing(path, smooth_factor, number_of_control_points)
            # Draw the path on the image
            annotated_image = draw_path_on_image(annotated_image, path)
            # Caclulate the turning rate
            height, width = image.shape[:2]
            robot_position_x = width // 2 # Assume the robot is in the middle of the image
            turning_rate = pid_controller.calculate_turning_rate(robot_position_x, path_position_x)
        else:
            turning_rate = None

        # Send the turning_rate to the robot
        send_turning_rate_to_robot(turning_rate)

        # Calculate FPS
        fps, prev_time = calculate_fps(prev_time)
        # Put the FPS on the image
        add_fps_on_image(annotated_image, fps)

        # Write the image into the output file
        out.write(annotated_image)

        # display the image
        cv2.imshow('Floor Detection and Path Planning', annotated_image)

        # Press 'e' to exit the loop and stop the video stream
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Release the video capture object and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

perception(video_source=2, output_file='predicted_output.mp4')
