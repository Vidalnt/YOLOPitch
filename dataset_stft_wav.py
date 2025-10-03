import os
import numpy as np
import torch
import librosa
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset
from scipy.io import wavfile
import config
import re

def _load_and_resample_audio(abs_wav_path, target_sr=16000):
    try:
        sr, audio = wavfile.read(abs_wav_path)
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    except Exception as e:
        print(f"Error loading {abs_wav_path}: {e}")
        return np.zeros(int(target_sr * 0.5))

def get_frames_and_stft_from_audio(audio, model_srate=16000, step_size=0.02, len_frame_time=0.064, n_fft=2047):
    try:
        hop_length = int(model_srate * step_size)
        wlen = int(model_srate * len_frame_time)
        n_frames = 1 + int((len(audio) - wlen) / hop_length)
        if n_frames <= 0:
            n_frames = 1
        total_len_needed = wlen + (n_frames - 1) * hop_length
        if len(audio) > total_len_needed:
            audio_proc = audio[:total_len_needed]
        elif len(audio) < total_len_needed:
            audio_proc = np.pad(audio, (0, total_len_needed - len(audio)), mode='constant', constant_values=0)
        else:
            audio_proc = audio

        frames = as_strided(audio_proc, shape=(wlen, n_frames),
                            strides=(audio_proc.itemsize, hop_length * audio_proc.itemsize))
        frames = frames.T.copy()
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        std = np.std(frames, axis=1)
        std[std == 0] = 1e-8
        frames /= std[:, np.newaxis]
        frames[np.isnan(frames)] = 0

        stft = librosa.stft(audio_proc,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=wlen,
                           window='hamming',
                           center=False)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        log_stft = log_stft.T

        assert frames.shape[0] == log_stft.shape[0], f"T mismatch: frames {frames.shape[0]}, stft {log_stft.shape[0]}"

        return frames.astype(np.float32), log_stft.astype(np.float32)
    except Exception as e:
        print(f"Error in get_frames_and_stft_from_audio: {e}")
        L = int(model_srate * len_frame_time)
        F = n_fft // 2 + 1
        T = 1
        return np.zeros((T, L), dtype=np.float32), np.zeros((T, F), dtype=np.float32)

class Net_DataSet(Dataset):
    def __init__(self, path):
        super(Net_DataSet, self).__init__()
        self.label = self.real_label(path)

    def real_label(self, path):
        with open(path, mode="r", encoding="gbk") as file:
            return file.readlines()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        every_file = self.label[index]
        parts = every_file.strip().split()

        if not parts:
            return self.__getitem__((index + 1) % self.__len__())

        if config.data_name == "ptdb":
            f0_rel_path = parts[0]
            audio_rel_path = f0_rel_path.replace("REF", "MIC").replace("ref", "mic").replace(".f0", ".wav")
            wav_path = os.path.join(config.wav_dir_path, audio_rel_path)
            pv_name = f0_rel_path
            label = parts[1:]
        else:
            if 'f0' in parts[0]:
                filename = parts[0].replace("ref", "mic").replace("f0", "txt")
                pv_name = filename.replace("txt", "f0").replace("mic", "ref")
            elif 'pv' in parts[0]:
                filename = parts[0].replace("pv", "txt")
                pv_name = filename.replace("txt", "pv")
            elif 'csv' in parts[0]:
                filename = parts[0].replace("csv", "txt")
                pv_name = filename.replace("txt", "csv")
            else:
                filename = parts[0]
                pv_name = parts[0]

            wavname = filename.replace("txt", "wav")
            wav_path = os.path.join(config.wav_dir_path, wavname)
            label = parts[1:]

        if not os.path.exists(wav_path):
            print(f"Audio file not found: {wav_path}")
            return self.__getitem__((index + 1) % self.__len__())

        audio = _load_and_resample_audio(wav_path, target_sr=config.sr if hasattr(config, 'sr') else 16000)
        frames, stft = get_frames_and_stft_from_audio(
            audio,
            model_srate=config.sr if hasattr(config, 'sr') else 16000,
            step_size=config.hop_size,
            len_frame_time=config.len_frame_time if hasattr(config, 'len_frame_time') else 0.064,
            n_fft=config.n_fft if hasattr(config, 'n_fft') else 2047
        )

        min_len = min(len(frames), len(stft), len(label))
        frames = frames[:min_len]
        stft = stft[:min_len]
        label = label[:min_len]

        label1 = []
        for x in label:
            try:
                x = int(np.float64(x))
                label1.append(x)
            except:
                label1.append(0)

        label1 = label1[:min_len]

        label1 = torch.tensor(label1).squeeze().long()
        frames = torch.tensor(frames).float().transpose(0, 1)
        stft = torch.tensor(stft).float().transpose(0, 1)

        return [frames, stft], [label1, pv_name]

if __name__ == "__main__":
    path = config.test_path
    s = Net_DataSet(path)
    print("s[6]:", s[6])
    print("frames shape:", s[6][0][0].shape)
    print("stft shape:", s[6][0][1].shape)
    print("label length:", len(s[6][1][0]))