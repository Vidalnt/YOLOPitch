import os
import numpy as np
from tqdm import tqdm
import config

def check_precomputed_features(precomputed_path):
    files = os.listdir(precomputed_path)
    frames_files = [f for f in files if f.endswith("_frames.npy")]
    stft_files = [f for f in files if f.endswith("_stft.npy")]
    
    frames_names = {f.replace("_frames.npy", "") for f in frames_files}
    stft_names = {f.replace("_stft.npy", "") for f in stft_files}
    
    common_names = frames_names & stft_names
    print(f"Found {len(common_names)} pairs of .npy files.")

    mismatched = []
    for name in tqdm(common_names, desc="Checking shapes"):
        frames_path = os.path.join(precomputed_path, f"{name}_frames.npy")
        stft_path = os.path.join(precomputed_path, f"{name}_stft.npy")

        frames = np.load(frames_path)
        stft = np.load(stft_path)

        if frames.ndim != 2:
            mismatched.append((name, f"frames.ndim = {frames.ndim}"))
            continue
        if stft.ndim != 2:
            mismatched.append((name, f"stft.ndim = {stft.ndim}"))
            continue

        if frames.shape[0] != stft.shape[0]:
            mismatched.append((name, f"T mismatch: {frames.shape[0]} vs {stft.shape[0]}"))

    if mismatched:
        print(f"Found {len(mismatched)} mismatched files:")
        for name, msg in mismatched:
            print(f"  {name}: {msg}")
    else:
        print("✅ All .npy files have correct shape and matching T dimension.")

if __name__ == "__main__":
    check_precomputed_features(config.precomputed_path)