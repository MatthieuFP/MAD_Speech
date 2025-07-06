import os
import argparse
from pprint import pprint
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import math
import torch
import torchaudio
from einops import rearrange

from model import MAD_Speech


def mel_transform(waveform, sample_rate=16_000):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        win_length=512,
        hop_length=256,
        n_mels=156
    )(waveform)
    return mel



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_audio", type=str, required=True, default="./audio_samples")
    return parser.parse_args()



def main(params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAD_Speech().to(device)

    # load weights
    state_dict = torch.load(hf_hub_download("matthieufp/mad_speech", "checkpoint.pt"), map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()

    # load data
    audio_files = os.listdir(params.path_audio)
    inputs = []
    for audio_file in tqdm(audio_files, "Reading Audio files..."):
        waveform, sample_rate = torchaudio.load(os.path.join(params.path_audio, audio_file))
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
        waveform = resampler(waveform)
        waveform = waveform[:, :96_000]  # First 6 seconds

        if waveform.size(1) < 96_000:  # Repeat if audio length < 6 seconds
            n_time_repeat = math.ceil(96_000 / waveform.size(1))
            waveform = torch.cat([waveform] * n_time_repeat, dim=-1)[:, :96_000]

        inputs.append(mel_transform(waveform))

    inputs = torch.cat(inputs, dim=0).to(device)
    inputs = rearrange(inputs, 'b d s -> b s d')
    with torch.inference_mode():
        mad_speech_scores = model(inputs)

    pprint(mad_speech_scores)


if __name__ == "__main__":
    params = get_parser()
    main(params)
