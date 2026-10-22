import os
import cv2
from ultralytics import YOLO

# 1. Load the trained model
model = YOLO(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best.pt'))

# 2. Open the CARLA video
video_path = 'parking_maneuver.mp4'
cap = cv2.VideoCapture(video_path)

# Read the source video properties for the output
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# 3. Output file setup
output_path = 'detection_result.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Run inference on the current frame
    # conf=0.5 filters out uncertain detections
    results = model(frame, conf=0.5, verbose=False)

    # 5. Draw the results on the frame
    # .plot() draws the bounding boxes and labels automatically
    annotated_frame = results[0].plot()

    # Save the processed frame
    out.write(annotated_frame)

    # (Optional) Show in real time when running on a local PC
    # cv2.imshow("YOLO11 CARLA Inference", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'): break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Video saved to: {output_path}")