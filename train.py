import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
from dataset_stft_wav import Net_DataSet
from tqdm import tqdm
from yolo_wav_stft import YoloBody
from formula_all import *
import logging
import config

log = feature.get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = "bf16"

train_dtype = torch.float32
use_amp = False

if precision == "bf16" and device == "cuda":
    if torch.cuda.is_bf16_supported():
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
scaler = GradScaler(device=device, enabled=use_scaler)

batch_size = 1
train_data = Net_DataSet(config.train_path)
validation_data = Net_DataSet(config.validation_path)
test_data = Net_DataSet(config.test_path)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size,
                                   num_workers=4, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             num_workers=4, pin_memory=True, persistent_workers=True)

model = YoloBody(phi='l', num_classes=config.out_class, pretrained=False).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

def get_label(path):
    pitch = []
    ref_cent = []
    with open(path, mode="r") as file:
        for line in file.readlines():
            if 'ptdb' in config.data_name:
                x = float(line.split(" ")[0])
                hz = x
                cent = Convert.convert_hz_to_cent(x) if x != 0 else 0
            elif 'ikala' in config.data_name or 'mir1k' in config.data_name:
                x = float(line.split(" ")[0])
                x = Convert.convert_semitone_to_hz(x)
                if x >= 10:
                    hz = x
                    cent = Convert.convert_hz_to_cent(x)
                else:
                    hz, cent = 0, 0
            elif 'mdb' in config.data_name:
                hz = float(line.split(",")[1].split("\n")[0])
                cent = Convert.convert_hz_to_cent(hz) if hz > 0 else 0
            pitch.append(hz)
            ref_cent.append(cent)
    return pitch, ref_cent

def train(dataloader, model, loss_fn, optimizer, scaler):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X_wav, X_stft = X[0].to(device), X[1].to(device)
        y = y[0].to(device).long()

        optimizer.zero_grad()

        with autocast(device, dtype=train_dtype, enabled=use_amp):
            pred = model(X_wav.unsqueeze(0), X_stft.unsqueeze(0))
            pred = pred.squeeze(0)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")

def validation(dataloader, model, loss_fn, csv_path=config.validation_csv_path):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = correct = total = 0
    voice_correct = voice_total = 0

    sound_score_list = []
    music_score_list = []

    with torch.no_grad():
        for X, Y in tqdm(dataloader):
            X_wav, X_stft = X[0].to(device), X[1].to(device)
            y = Y[0].to(device).long()
            filename = Y[1][0]

            with autocast(device, dtype=train_dtype, enabled=use_amp):
                pred = model(X_wav.unsqueeze(0), X_stft.unsqueeze(0)).squeeze(0)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
            total += y.numel()

            voiced_mask = (y != 0)
            voice_correct += (pred[voiced_mask].argmax(1) == y[voiced_mask]).sum().item()
            voice_total += voiced_mask.sum().item()

            pitch_data = pred.argmax(dim=1).cpu().numpy()
            pitch_hz = []
            for bin_val in pitch_data:
                if bin_val == 0:
                    pitch_hz.append(0.0)
                else:
                    if 'ptdb' in config.data_name:
                        pitch_hz.append(float(bin_val))
                    else:
                        cent = Convert.convert_bin_to_cent(bin_val)
                        hz = Convert.convert_cent_to_hz(cent)
                        pitch_hz.append(hz)

            try:
                ref_pitch, ref_cent = get_label(os.path.join(csv_path, filename))
            except:
                continue

            min_len = min(len(pitch_hz), len(ref_pitch))
            pitch_hz = np.array(pitch_hz[:min_len])
            ref_pitch = np.array(ref_pitch[:min_len])
            ref_cent = np.array(ref_cent[:min_len])
            pred_cent = np.array([Convert.convert_hz_to_cent(h) if h > 0 else 0 for h in pitch_hz[:min_len]])

            try:
                sound_score = Sound.all_mir_eval(pitch_hz, ref_pitch, threshold=0.2)
                music_score = Music.all_mir_eval(ref_pitch, ref_cent, pitch_hz, pred_cent, cent_tolerance=50)
                sound_score_list.append(sound_score)
                music_score_list.append(music_score)
            except:
                continue

    test_loss /= num_batches
    accuracy = correct / total if total > 0 else 0
    voice_acc = voice_correct / voice_total if voice_total > 0 else 0

    sound_avg = np.nanmean(np.array(sound_score_list), axis=0) if sound_score_list else [0]*6
    music_avg = np.nanmean(np.array(music_score_list), axis=0) if music_score_list else [0]*5

    print(f"Validation: Acc={100*accuracy:.1f}%, Voice={100*voice_acc:.1f}%, Loss={test_loss:.6f}")
    print(f"Sound: {sound_avg}, Music: {music_avg}")

    log.info(f"Val Acc: {100*accuracy:.1f}%, Voice: {100*voice_acc:.1f}%")
    log.info(f"Sound: {sound_avg}, Music: {music_avg}")

    return voice_acc

epochs = config.epochs
os.makedirs(config.save_model_path, exist_ok=True)
best_voice_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    log.info(f"Epoch {epoch+1}/{epochs}")

    train(train_dataloader, model, loss_fn, optimizer, scaler)
    current_acc = validation(validation_dataloader, model, loss_fn)

    if current_acc > best_voice_acc:
        best_voice_acc = current_acc
        best_path = os.path.join(config.save_model_path, "yolopitch_best.pth")
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved: {best_path} (Voice Acc: {best_voice_acc:.4f})")
        log.info(f"New best model saved: {best_path} (Voice Acc: {best_voice_acc:.4f})")

    if (epoch + 1) % config.save_interval == 0 or epoch == epochs - 1:
        ckpt_path = os.path.join(config.save_model_path, f"yolopitch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

final_path = os.path.join(config.save_model_path, "yolopitch_final.pth")
torch.save(model.state_dict(), final_path)
print("Training completed!")