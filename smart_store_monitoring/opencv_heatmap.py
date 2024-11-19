import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("models/yolov8n.pt")  # You can use a different variant if needed

# Open video capture (camera or video file)
cap = cv2.VideoCapture(
    "assets/video/people_01.mp4"
)  # Use 0 for webcam, or specify file path

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame size (width and height)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an empty heatmap to accumulate over time (same size as video frames)
heatmap = np.zeros((height, width), dtype=np.float32)

# Set up video writer to save output
output_filename = "outputs/video/people_opencv_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 file
out = cv2.VideoWriter(output_filename, fourcc, 30.0, (width, height))  # 30 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform object detection using YOLOv8
    results = model(frame)

    # Extract the bounding boxes for detected objects (results is a list now)
    # results[0] holds the detection result for the first image/frame
    detections = results[0].boxes  # Accessing the box data

    # Create a temporary heatmap for the current frame
    frame_heatmap = np.zeros_like(heatmap, dtype=np.float32)

    # Loop through detected objects and apply to heatmap
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates

        # Create a region of interest in the detected box
        frame_heatmap[y1:y2, x1:x2] += 1  # Increment heatmap intensity in detected area

    # Accumulate the frame heatmap over time
    heatmap += frame_heatmap

    # Normalize the heatmap to [0, 255] range
    normalized_heatmap = np.uint8(
        255 * heatmap / np.max(heatmap) if np.max(heatmap) > 0 else 1
    )

    # Apply a colormap to the heatmap for visualization
    colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original frame
    overlay = cv2.addWeighted(frame, 0.5, colored_heatmap, 0.5, 0)

    # Show the frame with overlay
    cv2.imshow("Heatmap Overlay", overlay)

    # Write the processed frame to the output video
    out.write(overlay)

    # Check for user input to break the loop (e.g., press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and writer, then close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
