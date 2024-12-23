import tkinter as tk
from tkinter import Tk
from text_to_speech import TextToSpeech
from object_detection import ObjectDetection
import cv2

# Main GUI setup
root = Tk()
root.bind("<Escape>", lambda e: root.quit())
root.attributes('-fullscreen', True)
root.title("YOLOv5 Object Detection")

# Initialize Object Detection and Text-to-Speech
tts = TextToSpeech()
detector = ObjectDetection()


# Function to start object detection and record the process
def start_detection():
    tts.speak("Starting object detection and recording.")

    frame_count = 0
    out = None

    try:
        while True:
            # Process frame through ObjectDetection
            frame, frame_count = detector.process_frame(frame_count, tts)

            if frame is not None:
                # Initialize video writer if not already done
                if out is None:
                    height, width, _ = frame.shape
                    out = cv2.VideoWriter(
                        'detection_output.avi',
                        cv2.VideoWriter_fourcc(*'XVID'),
                        20.0,
                        (width, height)
                    )

                # Write the frame to the output video
                out.write(frame)

                # Display the frame in a cv2 window
                cv2.imshow('Object Detection', frame)

            # Stop the process if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                tts.speak("Stopping object detection and saving the video.")
                break

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        tts.speak("An error occurred during object detection.")

    finally:
        # Release resources
        if out:
            out.release()
        detector.cap.release()
        cv2.destroyAllWindows()


# Main GUI layout
main_frame = tk.Frame(root, bg="white")
main_frame.place(relx=0.5, rely=0.5, width=2000, height=2000, anchor=tk.CENTER)

# Title and instructions
title_label = tk.Label(root, text="YOLOv5 Object Detection", font=('Rockwell', 20), bg="black", fg="white")
title_label.pack(side=tk.TOP, fill=tk.X)

exit_label = tk.Label(root, text="Press 'Q' to Quit", font=('Rockwell', 20), bg="black", fg="white")
exit_label.pack(side=tk.BOTTOM, fill=tk.X)

# Buttons
start_button = tk.Button(main_frame, text="Start Detection", command=start_detection, bg="gray", fg="black",
                         font=('Rockwell', 18))
start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Introductory message
tts.speak("Welcome to your eyes")

# Start the GUI main loop
root.mainloop()
