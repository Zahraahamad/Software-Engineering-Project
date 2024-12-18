from text_to_speech import TextToSpeech
from object_detection import ObjectDetection

def play_intro(tts):
    tts.speak("Welcome to your eyes.")

if __name__ == "__main__":
    tts = TextToSpeech()
    detector = ObjectDetection()

    play_intro(tts)
    print("[INFO] Starting Object Detection for Visually Impaired...")
    detector.run(tts)
