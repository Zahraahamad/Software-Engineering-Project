import torch
import pyttsx3
import cv2
import pandas as pd
import threading
import numpy as np
import time

# Constants for distance estimation
KNOWN_WIDTH = 14.0  # Known width of an object (e.g., in cm or inches)
FOCAL_LENGTH = 615.0  # Pre-calculated focal length of the camera (in pixels)

# Load a lighter version of the YOLOv5 model to reduce processing time
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', device='cpu', force_reload=True)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Open webcam (use 0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to convert object detection results to voice feedback asynchronously
def speak_detection(objects):
    def speak():
        for _, obj in objects.iterrows():
            label = obj['name']
            confidence = obj['confidence'] * 100
            width = obj['xmax'] - obj['xmin']
            distance = KNOWN_WIDTH * FOCAL_LENGTH / width  # Estimate distance
            message = (f"{label} detected with {confidence:.2f} percent confidence, "
                       f"approximately {distance:.2f} centimeters away.")
            engine.say(message)
        engine.runAndWait()

    # Start a new thread for the text-to-speech process
    threading.Thread(target=speak).start()

def play_intro():
    """Plays the intro message through text-to-speech."""
    engine.say("Welcome to your eyes.")
    engine.runAndWait()

def detect_color(frame):
    """Detects red regions in the frame and highlights them."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Function for real-time object detection
def detect_objects():
    """Runs the object detection loop."""
    print("[INFO] Object detection started. Press 'q' to stop.")
    frame_count = 0  # Initialize frame count to process every 2nd frame

    try:
        while True:
            ret, frame = cap.read()  # Capture frame from webcam
            if not ret:
                print("[ERROR] Failed to capture frame. Retrying...")
                time.sleep(0.5)
                continue

            if frame_count % 2 == 0:
                results = model(frame)
                detected_objects = results.pandas().xyxy[0]  # Get detected objects as a pandas dataframe

                if not detected_objects.empty:
                    speak_detection(detected_objects)  # Notify the user of detected objects
                    for _, obj in detected_objects.iterrows():
                        # Draw bounding boxes and display distance
                        x_min, y_min, x_max, y_max = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
                        label = obj['name']
                        width = obj['xmax'] - obj['xmin']
                        distance = KNOWN_WIDTH * FOCAL_LENGTH / width  # Estimate distance
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        cv2.putText(frame, f"{label}: {distance:.2f} cm", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                frame = detect_color(frame)

            # Display the frame with detection
            cv2.imshow('Object Detection', frame)
            frame_count += 1

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting object detection.")
                break

    except Exception as e:
        print(f"[ERROR] Unexpected error in main loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Play the intro message
    play_intro()

    # Start object detection
    print("[INFO] Starting Object Detection for Visually Impaired...")
    detect_objects()
