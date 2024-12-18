import pyttsx3
import threading


class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def speak(self, message):
        self.engine.say(message)
        self.engine.runAndWait()

    def speak_async(self, messages):
        def speak():
            for message in messages:
                self.engine.say(message)
            self.engine.runAndWait()

        threading.Thread(target=speak).start()
