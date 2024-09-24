import pyaudio
import numpy as np
import whisper
import time
from pynput import keyboard

# Load ASR model
asr_model = whisper.load_model("tiny.en")

def record_audio():
    recording = False

    # Keyboard event handling for starting/stopping recording
    def on_press(key):
        nonlocal recording
        if key == keyboard.Key.shift:
            recording = True

    def on_release(key):
        nonlocal recording
        if key == keyboard.Key.shift:
            recording = False
            return False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print('Press Shift to start recording...')
    while not recording:
        time.sleep(0.1)  # Wait for recording to start

    print('Recording started...')
    
    # Initialize PyAudio stream for recording
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=1024, input=True)
    frames = []

    # Record audio until Shift is released
    while recording:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print('Recording stopped.')
    
    # Close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Return recorded audio as a single NumPy array
    return np.hstack(frames)

def transcribe_audio(audio_data):
    # Ensure audio_data is a float32 NumPy array
    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize audio data
    result = asr_model.transcribe(audio_data)  # Transcribe normalized audio
    return result['text']

# Example usage:
def main():
    audio_data = record_audio()  # Record audio
    transcription = transcribe_audio(audio_data)  # Transcribe recorded audio
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()
