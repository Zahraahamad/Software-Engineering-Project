import pyttsx3
import threading
import time


class TextToSpeech:
    def __init__(self):
        #initialization
        self.engine = pyttsx3.init()
    #speaking speed
        self.engine.setProperty('rate', 140)

    #speaks a single message and blocks the program until the message has been completely spoken.
    def speak(self, message):
        self.engine.say(message)
        self.engine.runAndWait()
        time.sleep(0.2)

    #This function is an asynchronous implementation of
    # text-to-speech. It speaks a list of messages
    # without blocking the main program.
    # It achieves this by defining an inner speak function and running it in a separate thread.
    def speak_async(self, messages):
        def speak():
            for message in messages:
                self.engine.say(message)
            self.engine.runAndWait()


        threading.Thread(target=speak).start()
