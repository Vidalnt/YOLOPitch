import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataset_stft_wav import get_frames, get_stft
import config
from tqdm import tqdm

def precompute_features():
    os.makedirs(config.precomputed_path, exist_ok=True)
    
    audio_files = []
    for root, _, files in os.walk(config.wav_dir_path):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                if config.data_name == 'ptdb' and os.path.sep + 'LAR' + os.path.sep in audio_path:
                    continue
                audio_files.append(audio_path)
    
    for audio_path in tqdm(audio_files, desc="Precomputing features"):
        frames = get_frames(audio_path, step_size=config.hop_size)  # (T, L)
        stft = get_stft(audio_path, step_size=config.hop_size).T    # (T, F)
        
        if frames.ndim != 2:
            print(f"⚠️  frames.ndim != 2: {frames.shape} for {audio_path}")
            continue
        if stft.ndim != 2:
            print(f"⚠️  stft.ndim != 2: {stft.shape} for {audio_path}")
            continue

        if frames.shape[0] != stft.shape[0]:
            print(f"⚠️  T mismatch: frames {frames.shape[0]}, stft {stft.shape[0]} for {audio_path}")
            continue

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        np.save(os.path.join(config.precomputed_path, f"{base_name}_frames.npy"), frames)
        np.save(os.path.join(config.precomputed_path, f"{base_name}_stft.npy"), stft)

if __name__ == "__main__":
    precompute_features()