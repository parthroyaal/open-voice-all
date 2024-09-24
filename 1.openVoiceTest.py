import pyaudio
import torch
import os
import re
from utils import split_sentences_latin
from api import BaseSpeakerTTS, ToneColorConverter

# Path setup
tts_en_ckpt_base = os.path.join(os.path.dirname(__file__), "checkpoints/base_speakers/EN")
tts_ckpt_converter = os.path.join(os.path.dirname(__file__), "checkpoints/converter")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load TTS model and converter
tts_model = BaseSpeakerTTS(f'{tts_en_ckpt_base}/config.json', device=device)
tts_model.load_ckpt(f'{tts_en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{tts_ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{tts_ckpt_converter}/checkpoint.pth')
sampling_rate = tts_model.hps.data.sampling_rate
mark = tts_model.language_marks.get("english", None)

# Simplified play_audio function
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

# Test TTS system by speaking "Hello, this is a test."
play_audio("Hello, this is a test.")
