from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for YOLOv8 nano model

# Start video capture (use 0 for webcam or replace with a video file path)
cap = cv2.VideoCapture("./1.mp4")

# Define the desired resolution (e.g., 640x480)
target_resolution = (1280, 960)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply frame modifications (example: convert to grayscale and apply Gaussian blur)
    modified_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    modified_frame = cv2.GaussianBlur(modified_frame, (5, 5), 0)  # Apply Gaussian blur

    # Resize the original frame to the target resolution
    resized_frame = cv2.resize(frame, target_resolution)

    # Use the YOLO model to detect objects on the original frame (not modified)
    results = model(resized_frame)

    # Iterate over each detection result
    for result in results[0].boxes:  # Iterate over all detections
        if result.cls == 0:  # Class ID 0 corresponds to 'person'
            x1, y1, x2, y2 = result.xyxy[0]  # Get bounding box coordinates
            conf = result.conf[0]  # Confidence score

            # Draw bounding box on the resized frame
            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(resized_frame, f'Person {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the modified frame
    cv2.imshow('Modified Frame', modified_frame)  # This shows the modified frame (grayscale + blur)
    
    # Display the resized frame with detections
    cv2.imshow('YOLOv8 Person Detection', resized_frame)  # This shows the resized frame with bounding boxes

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
