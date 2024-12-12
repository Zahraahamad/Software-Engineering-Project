import torch
import pyttsx3
import cv2
import pandas as pd
import threading

# Load a lighter version of the YOLOv5 model to reduce processing time
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu', force_reload=True)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Open webcam (use 0 for default webcam)
cap = cv2.VideoCapture(0)


# Function to convert object detection results to voice feedback asynchronously
def speak_detection(objects):
    def speak():
        for _, obj in objects.iterrows():
            label = obj['name']
            confidence = obj['confidence'] * 100
            message = f"{label} detected with {confidence:.2f} percent confidence."
            engine.say(message)
        engine.runAndWait()

    # Start a new thread for the text-to-speech process
    threading.Thread(target=speak).start()


# Function for real-time object detection
def detect_objects():
    frame_count = 0  # Initialize frame count to process every 2nd frame

    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            break

        # Process every second frame to reduce processing load
        if frame_count % 2 == 0:
            results = model(frame)
            detected_objects = results.pandas().xyxy[0]  # Get detected objects as a pandas dataframe

            if not detected_objects.empty:
                speak_detection(detected_objects)  # Notify the user of detected objects

        # Display the frame with detection (only for debugging purposes)
        cv2.imshow('Object Detection', frame)
        frame_count += 1

        # Exit the loop if 'q' is pressed (for debugging, not for final app)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Object Detection for Visually Impaired...")
    detect_objects()