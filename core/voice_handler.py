import pyttsx3
import threading

class VoiceSystem:
    def __init__(self):
        # We don't initialize the engine here to avoid threading conflicts.
        # We initialize it inside the thread every time we speak.
        pass

    def speak(self, text):
        """Public method to trigger speech without freezing the app"""
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        """Internal thread logic"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 230)  # Speed (Default is usually 200)
            engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)
            
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Voice Error: {e}")