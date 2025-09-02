import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import librosa
import os
import re
import sys
from dataset_stft_wav import Net_DataSet
from tqdm import tqdm
from yolo_wav_stft import YoloBody
from formula_all import *
import time
import logging
import config

log = feature.get_logger()

precision = "bf16"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dtype = torch.float32
use_amp = False

if precision == "bf16" and device == "cuda":
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        train_dtype = torch.bfloat16
        use_amp = True
        print("Training configured for BFloat16 (bf16).")
    else:
        print("NOTE: Current GPU does not support bf16. fp32 will be used instead.")
elif precision == "fp16" and device == "cuda":
    train_dtype = torch.float16
    use_amp = True
    print("Training configured for Float16 (fp16).")
else:
    print("Training configured for Float32 (fp32).")

use_scaler = (precision == "fp16") and (device == "cuda")
scaler = GradScaler(enabled=use_scaler)

validation_csv_path = config.validation_csv_path
train_data = Net_DataSet(config.train_path)
validation_data = Net_DataSet(config.validation_path)
test_data = Net_DataSet(config.test_path)
batch_size = 1
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True, persistent_workers=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

resume_path = ""
model = YoloBody(phi='l', pretrained=False).to(device)
if resume_path:
    model.load_state_dict(torch.load(resume_path))
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

def get_label(path):
    pitch = []
    ref_cent = []
    with open(path, mode="r") as file:
        for line in file.readlines():
            if 'ptdb' in config.data_name :
                x = float(line.split(" ")[0])
                hz = x
                cent = Convert.convert_hz_to_cent(x) if x != 0 else 0
            elif 'ikala' in config.data_name or  'mir1k' in config.data_name:
                x = float(line.split(" ")[0])
                x = Convert.convert_semitone_to_hz(x)
                if x >= 10:
                    hz = x
                    cent = Convert.convert_hz_to_cent(x)
                else:
                    hz, cent = 0, 0
            elif 'mdb' in config.data_name :
                hz = float(line.split(",")[1].split("\n")[0])
                cent = Convert.convert_hz_to_cent(hz) if hz > 0 else 0
            pitch.append(hz) 
            ref_cent.append(cent)               
    return pitch,ref_cent

def train(dataloader, model, loss_fn, optimizer, scaler):
    size = len(dataloader.dataset) 
    model.train()

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        y = y[0].type(torch.LongTensor)
        X_wav, X_stft, y = X[0].to(device), X[1].to(device), y.to(device).squeeze(0)
        
        optimizer.zero_grad()

        with autocast(device_type=device, dtype=train_dtype, enabled=use_amp):
            pred = model(X_wav, X_stft)
            min_num = min(y.shape[0], pred.shape[0])
            pred = pred[:min_num, :]
            y = y[:min_num]
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss_item, current = loss.item(), batch * len(X)
            print(f"loss: {loss_item:>7f}  [{current:>5d}/{size:>5d}]")

def validation(dataloader, model, loss_fn, csv_path=validation_csv_path):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    total, voice_total, voice_correct = 0, 0, 0
    
    with torch.no_grad():
        sound_score_list = []
        music_score_list = []
        for X, Y in dataloader:
            y = Y[0].type(torch.LongTensor)
            X_wav, X_stft, y = X[0].to(device), X[1].to(device), y.to(device).squeeze(0)

            with autocast(device_type=device, dtype=train_dtype, enabled=use_amp):
                pred = model(X_wav, X_stft)

            min_num = min(y.shape[0], pred.shape[0])
            pred = pred[:min_num, :]
            y = y[:min_num]
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            total += y.shape[0]
            zero_index = y != 0
            voice_correct += (pred[zero_index].argmax(1) == y[zero_index]).type(torch.float).sum().item()
            voice_total += zero_index.sum().item()
            
            pitch_data = pred.argmax(dim=1).cpu().numpy()
            pitch_cent = []
            pitch = []
            
            for i in pitch_data:
                predict_hz, predict_cent = 0, 0
                if i != 0:
                    if config.data_name in 'ptdb':
                        predict_hz = i
                        predict_cent = Convert.convert_hz_to_cent(i)
                    else:
                        predict_cent = Convert.convert_bin_to_cent(i)  
                        predict_hz = Convert.convert_cent_to_hz(predict_cent)
                pitch.append(predict_hz)
                pitch_cent.append(predict_cent)

            ref_cent_list, label_data = get_label(os.path.join(csv_path, Y[1][0]))

            min_len = min(len(pitch), len(label_data))
            pitch = np.array(pitch[:min_len])
            label = np.array(label_data[:min_len])
            pitch_cent = np.array(pitch_cent[:min_len])
            label_cent = np.array(ref_cent_list[:min_len])

            sound_score = Sound.all_mir_eval(pitch, label, threshold=0.2)
            music_score = Music.all_mir_eval(label, label_cent, pitch, pitch_cent, cent_tolerance=50)
            sound_score_list.append(sound_score)
            music_score_list.append(music_score)
    
    test_loss /= num_batches
    accuracy = correct / total if total > 0 else 0
    voice_accuracy = voice_correct / voice_total if voice_total > 0 else 0
    
    print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Voice Correct: \n Voice Accuracy: {(100*voice_accuracy):>0.1f}% \n")

    sound_score_numpy = np.array(sound_score_list)
    music_score_numpy = np.array(music_score_list)
    sound_score_avg = np.nanmean(sound_score_numpy, axis=0)
    music_score_avg = np.nanmean(music_score_numpy, axis=0)
    
    print("Score:")
    print("sound_score_avg:", sound_score_avg)
    print('music_score_avg:', music_score_avg)
    
    log.info(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    log.info(f"Voice Correct: \n Voice Accuracy: {(100*voice_accuracy):>0.1f}% \n")
    log.info("Score:")
    log.info(f"sound_score_avg: {sound_score_avg}")
    log.info(f"music_score_avg: {music_score_avg}")

epochs = config.epochs
save_model_path = 'save_model_path'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    log.info(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, scaler)
    validation(validation_dataloader, model, loss_fn)
    log.info(f"\n\n")
    validation(test_dataloader, model, loss_fn, csv_path=config.test_csv_path)

print("Done!")