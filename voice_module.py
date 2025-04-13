
# ======================== voice_module.py ========================
from gtts import gTTS
from playsound import playsound
import os

def speak(text, lang="th"):
    tts = gTTS(text=text, lang=lang)
    tts.save("temp.mp3")
    playsound("temp.mp3")
    os.remove("temp.mp3")