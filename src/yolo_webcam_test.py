# yolo_webcam_test.py

from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8l.pt")

# Open DJI webcam via V4L2
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Fallback if camera not opened
if not cap.isOpened():
    print("Camera not opened - trying index 0")
    cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame grab failed")
        break

    # Run YOLO inference
    results = model.predict(
        source=frame,
        imgsz=640,
        device=0,
        verbose=False
    )

    # Annotate frame
    annotated = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 - DJI Webcam", annotated)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
