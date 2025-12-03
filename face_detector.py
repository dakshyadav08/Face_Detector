import cv2
import os
import sys
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION AND INITIALIZATION
# ==============================================================================

# --- File Paths ---
# The 'r' prefix creates a raw string, which reliably handles Windows backslashes.
INPUT_VIDEO_PATH = r'E:\Sample\Internship\test_video.mp4'
OUTPUT_VIDEO_PATH = 'output_video.mp4'
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for a detection (0.0 to 1.0)

# Check if the input file exists before proceeding
if not os.path.exists(INPUT_VIDEO_PATH):
    print(f"Error: Input video file not found at '{INPUT_VIDEO_PATH}'")
    sys.exit(1)


# 2. Load the YOLOv8 Face Detection Model
# NOTE: Using the file name provided by the user: 'yolov8n_100e.pt'
try:
    # This line attempts to load the model from the local file: yolov8n_100e.pt
    yolo_model = YOLO('yolov8n_100e.pt')
    print("YOLOv8 Face model loaded successfully.")
except Exception as e:
    # Error handling for the model load failure
    if "No such file or directory" in str(e) or "FileNotFoundError" in str(e):
        print("\n--- Model Load Error ---")
        print("ACTION REQUIRED: The file 'yolov8n_100e.pt' was not found.")
        print("Please ensure 'yolov8n_100e.pt' is in the same directory as this script.")
    else:
        print(f"Error loading YOLO model: {e}")
    sys.exit(1)


# ==============================================================================
# 3. VIDEO PROCESSING SETUP
# ==============================================================================
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video stream from {INPUT_VIDEO_PATH}")
    sys.exit(1)

# Get video properties (width, height, FPS) for the output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print(f"Starting processing: {frame_width}x{frame_height} @ {fps} FPS")
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ==============================================================================
# 4. MAIN FRAME PROCESSING LOOP
# ==============================================================================
while cap.isOpened():
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break # End of video or error reading frame

    # --- PROGRESS TRACKING (Optional, but helpful for long videos) ---
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processing frame {frame_count}/{total_frames}...")

    # Perform inference (Face Detection)
    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # 5. Draw Bounding Boxes
    for result in results:
        # Get bounding boxes in (x1, y1, x2, y2) format
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            x1, y1, x2, y2 = box

            # Draw the bounding box (Green BGR color: (0, 255, 0))
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Add the 'Face' label
            label = "Face"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Position the text slightly above the box
            text_pos = (x1, y1 - 10 if y1 > 20 else y1 + 25) 
            cv2.putText(frame, label, text_pos, font, font_scale, color, font_thickness)
            
    # 6. Write the processed frame to the output video
    out.write(frame)


# ==============================================================================
# 7. CLEANUP
# ==============================================================================
cap.release()
out.release()
cv2.destroyAllWindows()
print("-" * 50)
print(f"Processing complete! {frame_count} frames processed.")
print(f"Output saved to: {os.path.abspath(OUTPUT_VIDEO_PATH)}")
print("-" * 50)