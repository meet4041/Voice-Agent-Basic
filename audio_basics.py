import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION = 5

print("Start speaking...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording finished.")

write("my_voice.wav", SAMPLE_RATE, audio)

audio = audio.flatten()

time = np.linspace(
    0,
    DURATION,
    len(audio)
)

plt.plot(time, audio)

plt.title("My Voice Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.show()