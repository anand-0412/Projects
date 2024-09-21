import cv2
import cvzone
import math
import numpy as np
from sort import *
from ultralytics import YOLO

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

# Load the YOLO model
model = YOLO("yolov8l.pt")

# Define the classes YOLO can detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
total_count = []

# Define region of interest (ROI) as the center region
# For example, take the center region as the middle 50% of the frame
frame_width = 1280
frame_height = 720
roi_x1, roi_y1 = int(frame_width * 0.35), int(frame_height * 0.11)  # Top-left corner of the center region
roi_x2, roi_y2 = int(frame_width * 0.60), int(frame_height * 0.90)  # Bottom-right corner of the center region

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform object detection with YOLO
    results = model(img, stream=True)
    detections = np.empty((0, 5))

    # Process the results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            currentClass = classNames[cls]

            # Check if detected object is a bottle with confidence > 0.4
            if currentClass == "bottle" and conf > 0.4:
                # Check if the bottle is within the center ROI
                bottle_center_x = (x1 + x2) // 2
                bottle_center_y = (y1 + y2) // 2

                if roi_x1 < bottle_center_x < roi_x2 and roi_y1 < bottle_center_y < roi_y2:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

    # Track the detections
    resultsTracker = tracker.update(detections)

    # Draw the ROI rectangle (center region)
    cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)

    # Draw bounding boxes and tracker IDs
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

        # If this is a new object, add it to total count
        if total_count.count(Id) == 0:
            total_count.append(Id)

    # Display the count of detected bottles
    cv2.putText(img, str(len(total_count)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Show the output frame
    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
