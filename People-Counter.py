from ultralytics import YOLO  # Import YOLO model for object detection
import cv2  # Import OpenCV library for image processing
import cvzone  # Import cvzone library for drawing bounding boxes and text
import math  # Import math library for mathematical operations
import time  # Import time library for time-related operations
from sort import Sort  # Import Sort class for object tracking
import numpy as np  # Import numpy library for array manipulation
import os  # Import os library for file and directory operations

# Open the video file for processing
cap = cv2.VideoCapture("../Videos/people.mp4")

# Load YOLO model with the specified weights file
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Output folder for saving frames
output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)

# Initialize SORT tracker with specified parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define region limits for counting
limitsUP = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []

# Define class names for objects that can be detected
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

# Load mask image
mask = cv2.imread('mask.png')

# Main loop for processing each frame of the video
while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        break  # Exit the loop if there are no more frames to read

    # Apply mask to restrict detection to a specific region
    imgRegion = cv2.bitwise_and(img, mask)

    # Overlay graphics image on the frame
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    # Perform object detection with YOLO on the current frame region
    results = model(imgRegion, stream=True)

    # Initialize an empty array to store detections
    detections = np.empty((0, 5))

    # Process the results of object detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract box coordinates, confidence, and class
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Get the class name
            currentClass = classNames[cls]

            # Check if the detected object is a person with high confidence
            if currentClass == 'person' and conf > 0.3:
                # Store the detection in the array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with detections
    resultsTracker = tracker.update(detections)

    # Draw region limits and track objects
    cv2.line(img, (limitsUP[0], limitsUP[1]), (limitsUP[2], limitsUP[3]), (0, 255, 0), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box and ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if objects cross the counting lines and update counts
        if limitsUP[0] < cx < limitsUP[2] and limitsUP[1] - 15 < cy < limitsUP[1] + 15:
            if id not in totalCountUp:
                totalCountUp.append(id)
                cv2.line(img, (limitsUP[0], limitsUP[1]), (limitsUP[2], limitsUP[3]), (0, 0, 255), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if id not in totalCountDown:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # Display counts on the frame
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    # Save frame with YOLO detections applied
    output_path = os.path.join(output_folder, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}.jpg")
    cv2.imwrite(output_path, img)

    # Display the processed frame
    cv2.imshow("Image", img)

    # Wait for a key press to move to the next frame
    cv2.waitKey(1)

# Release video capture
cap.release()

# Recreate video from saved frames
output_video_path = "output_video.mp4"
img_array = []

# Iterate through saved frames and append to img_array
for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".jpg"):
        img_path = os.path.join(output_folder, filename)
        img = cv2.imread(img_path)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

# Write the frames to a video file
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

print("Video created successfully.")

