import pyaudio
import wave
import os
import numpy as np
import io
import time
from faster_whisper import WhisperModel
import re
import torch
from utils import split_sentences_latin
from api import BaseSpeakerTTS, ToneColorConverter

# Setup TTS paths and models
tts_en_ckpt_base = os.path.join(os.path.dirname(__file__), "checkpoints/base_speakers/EN")
tts_ckpt_converter = os.path.join(os.path.dirname(__file__), "checkpoints/converter")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tts_model = BaseSpeakerTTS(f'{tts_en_ckpt_base}/config.json', device=device)
tts_model.load_ckpt(f'{tts_en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{tts_ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{tts_ckpt_converter}/checkpoint.pth')
sampling_rate = tts_model.hps.data.sampling_rate
mark = tts_model.language_marks.get("english", None)

# Function to play audio using OpenVoice TTS
def play_audio(text):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sampling_rate, output=True)
    texts = split_sentences_latin(text)
    for t in texts:
        audio_list = []
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        t = f'[{mark}]{t}[{mark}]'
        stn_tst = tts_model.get_text(t, tts_model.hps, False)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(tts_model.device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(tts_model.device)
            sid = torch.LongTensor([tts_model.hps.speakers["default"]]).to(tts_model.device)
            audio = tts_model.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.6)[0][0, 0].data.cpu().float().numpy()
            audio_list.append(audio)
        data = tts_model.audio_numpy_concat(audio_list, sr=sampling_rate).tobytes()
        stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to calculate volume for silence detection
def calculate_volume(audio_data):
    return np.abs(audio_data).mean()

def is_silent(audio_data, threshold):
    volume = calculate_volume(audio_data)
    return volume < threshold, volume

# Function to record and check volume before transcription
def record_and_transcribe(p, stream, model, chunk_length=3, volume_threshold=300, transcription_threshold=100):
    frames = []
    silent_chunks = 0
    max_silent_chunks = 10  # Number of consecutive silent chunks before stopping
    
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        is_silent_chunk, volume = is_silent(audio_data, transcription_threshold)
        print(f"Current volume: {volume:.2f}", end="\r")
        
        if volume > volume_threshold:  # Only process if volume is above threshold
            if not is_silent_chunk:
                frames.append(data)
                silent_chunks = 0
            else:
                silent_chunks += 1
                if frames and silent_chunks >= max_silent_chunks:
                    break
        else:
            print(f"Volume too low ({volume:.2f}), skipping transcription...\r", end="")
            return None
    
    if not frames:
        return None
    
    # Convert frames to in-memory WAV file
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
        wav_buffer.seek(0)
        
        # Transcribe the audio
        segments, _ = model.transcribe(wav_buffer, language="en")
        transcription = " ".join([segment.text for segment in segments])
    
    return transcription

def main():
    # Initialize Whisper model
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    
    print("Listening and transcribing... (Press Ctrl+C to stop)")
    print("Speak into your microphone. Adjust the threshold if needed.")
    
    volume_threshold = 300  # Minimum volume required to start transcription
    transcription_threshold = 100  # Threshold for silence detection during transcription
    
    try:
        while True:
            transcription = record_and_transcribe(p, stream, model, volume_threshold=volume_threshold, transcription_threshold=transcription_threshold)
            
            if transcription:
                print(f"\nTranscription: {transcription}")
                if "hi" in transcription.lower():
                    print("Detected 'hi'. Responding with 'Yes boss'...")
                    play_audio("Yes boss")
                    time.sleep(2)  # Add a delay to prevent immediate re-transcription of the response
            else:
                print("\nListening... (no significant audio detected)", end="\r")
    
    except KeyboardInterrupt:
        print("\nStopped listening.")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
