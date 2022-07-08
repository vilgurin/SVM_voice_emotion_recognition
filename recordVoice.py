import sounddevice
from scipy.io.wavfile import write
def record():
    fps = 16000
    duration = 4
    print("Recoding...")

    recording = sounddevice.rec((duration*fps), samplerate=fps, channels=1)

    sounddevice.wait()
    print("Done!")

    write("input.wav", fps, recording)
