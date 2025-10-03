import os
import numpy as np
import torch
import librosa
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset
from scipy.io import wavfile
import config
import re

def get_frames(abs_wav_path, model_srate=16000, step_size=0.02, len_frame_time=0.064):
    try:
        sample_rate, audio = wavfile.read(abs_wav_path)
        audio = audio.astype(np.float32)
        
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        hop_length = int(sample_rate * step_size)  
        wlen = int(sample_rate * len_frame_time)
        
        if sample_rate != model_srate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=model_srate)
            sample_rate = model_srate
            hop_length = int(sample_rate * step_size)
            wlen = int(sample_rate * len_frame_time)

        n_frames = 1 + int((len(audio) - wlen) / hop_length)
        
        if n_frames <= 0:
            return np.zeros((1, wlen), dtype=np.float32)
        
        total_len_needed = wlen + (n_frames - 1) * hop_length
        if len(audio) > total_len_needed:
            audio = audio[:total_len_needed]
        elif len(audio) < total_len_needed:
            audio = np.pad(audio, (0, total_len_needed - len(audio)), mode='constant')

        frames = as_strided(audio, shape=(wlen, n_frames),
                            strides=(audio.itemsize, hop_length * audio.itemsize))
        frames = frames.T.copy()
        
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        std = np.std(frames, axis=1)
        std[std == 0] = 1e-8
        frames /= std[:, np.newaxis]
        frames[np.isnan(frames)] = 0
        
        return frames
    except Exception as e:
        print(f"Error processing {abs_wav_path}: {e}")
        wlen = int(model_srate * len_frame_time)
        return np.zeros((1, wlen), dtype=np.float32)

def get_stft(abs_wav_path, model_srate=16000, step_size=0.02, n_fft=2047, len_frame_time=0.064):
    try:
        y, sr = librosa.load(abs_wav_path, sr=model_srate, mono=True)
        
        hop_length = int(model_srate * step_size)
        wlen = int(model_srate * len_frame_time)
        
        n_frames = 1 + int((len(y) - wlen) / hop_length)

        total_len_needed = wlen + (n_frames - 1) * hop_length
        if len(y) > total_len_needed:
            y = y[:total_len_needed]
        elif len(y) < total_len_needed:
            y = np.pad(y, (0, total_len_needed - len(y)), mode='constant')

        stft = librosa.stft(y, n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=wlen,
                           window='hamming',
                           center=False)
        log_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return log_stft
    except Exception as e:
        print(f"Error processing STFT of {abs_wav_path}: {e}")
        n_freq_bins = n_fft // 2 + 1
        return np.zeros((n_freq_bins, 10), dtype=np.float32)


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
        
        try:
             base_name = os.path.splitext(wavname)[0]
             frames_path = os.path.join(config.precomputed_path, f"{base_name}_frames.npy")
             stft_path = os.path.join(config.precomputed_path, f"{base_name}_stft.npy")
             frames = np.load(frames_path)
             stft = np.load(stft_path)
        except:
            frames = get_frames(wav_path, step_size=config.hop_size)
            stft = get_stft(wav_path, step_size=config.hop_size, len_frame_time=config.len_frame_time).T

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