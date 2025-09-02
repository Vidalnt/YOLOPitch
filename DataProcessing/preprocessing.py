import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataset_stft_wav import get_frames, get_stft
import config

def precompute_features():
    os.makedirs(config.precomputed_path, exist_ok=True)
    
    audio_files = []
    for root, _, files in os.walk(config.wav_dir_path):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    
    for audio_path in audio_files:
        print(f"Processing: {audio_path}")
        
        frames = get_frames(audio_path, step_size=config.hop_size)
        stft = get_stft(audio_path, step_size=config.hop_size).T
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        np.save(os.path.join(config.precomputed_path, f"{base_name}_frames.npy"), frames)
        np.save(os.path.join(config.precomputed_path, f"{base_name}_stft.npy"), stft)

if __name__ == "__main__":
    precompute_features()