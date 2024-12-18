import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from config import KNOWN_WIDTH, FOCAL_LENGTH

class ObjectDetection:
    def __init__(self, model_path='yolov8x.pt'):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def calculate_distance(self, width):
        return KNOWN_WIDTH * FOCAL_LENGTH / width

    def detect_color(self, frame):
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

    def process_frame(self, frame_count, tts):
        ret, frame = self.cap.read()
        if not ret:
            tts.speak("Failed to capture frame.")
            time.sleep(0.5)
            return None, frame_count

        if frame_count % 2 == 0:
            results = self.model(frame)
            detected_objects = results[0].boxes.data.cpu().numpy()

            messages = []
            for box in detected_objects:
                x_min, y_min, x_max, y_max, confidence, class_id = box[:6]
                width = x_max - x_min
                distance = self.calculate_distance(width)
                class_name = self.model.names.get(int(class_id), "Unknown")

                messages.append(f"{class_name} detected approximately {distance:.2f} centimeters away.")

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name}: {distance:.2f} cm", (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if messages:
                tts.speak_async(messages)

            frame = self.detect_color(frame)

        frame_count += 1
        return frame, frame_count

    def run(self, tts):
        print("[INFO] Object detection started. Press 'q' to stop.")
        frame_count = 0

        try:
            while True:
                frame, frame_count = self.process_frame(frame_count, tts)
                if frame is not None:
                    cv2.imshow('Object Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Exiting object detection.")
                    break

        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
