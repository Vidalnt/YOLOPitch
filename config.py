import numpy as np
import math
import warnings
import torch
import time
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import librosa
import time
import logging
import os

notice=''

data_name = 'ptdb'  #choose: mdb,ptdb,ikala,mir1k,mdb_zi,mdb_song

epochs=100
save_interval = 2
save_model_path = '/kaggle/working/model'
precomputed_path = '/kaggle/tmp'
data_dict={

    'ptdb_train_path' : '/kaggle/working/SPEECH_DATA/SPEECH DATA/label/hz_class_2/train_label.ark',
    'ptdb_validation_path' : '/kaggle/working/SPEECH_DATA/SPEECH DATA/label/hz_class_2/cv_label.ark',
    'ptdb_validation_csv_path' : '/kaggle/working/SPEECH_DATA/SPEECH DATA/',
    'ptdb_test_path' : '/kaggle/working/SPEECH_DATA/SPEECH DATA/label/hz_class_2/test_label.ark',
    'ptdb_wav_dir_path' : '/kaggle/working/SPEECH_DATA/SPEECH DATA',
    'ptdb_hop_size' : 0.01,
    'ptdb_hop_size_dot' : 0.01*16000,
    'ptdb_out_class' : 360,
    'ptdb_sr' : 16000,

    'mir1k_train_path' : '/kaggle/working/MIR-1K/data/label/bin_class/train_label.ark',
    'mir1k_validation_path' : '/kaggle/working/MIR-1K/data/label/bin_class/cv_label.ark',
    'mir1k_validation_csv_path' : '/kaggle/working/MIR-1K/PitchLabel',
    'mir1k_test_path' : '/kaggle/working/MIR-1K/data/label/bin_class/test_label.ark',
    'mir1k_wav_dir_path' : '/kaggle/working/MIR-1K/Wavfile',
    'mir1k_hop_size' : 0.02,
    'mir1k_hop_size_dot' : 0.02*16000,
    'mir1k_out_class' : 360,
    'mir1k_sr' : 16000,


    'mdb_train_path' : '/kaggle/working/MDB-stem-synth/data/all_label/train_label.ark',
    'mdb_validation_path' : '/kaggle/working/MDB-stem-synth/data/all_label/cv_label.ark',
    'mdb_validation_csv_path' : '/kaggle/working/MDB-stem-synth/annotation_stems',
    'mdb_test_path' : '/kaggle/working/MDB-stem-synth/data/all_label/test_label.ark',
    'mdb_wav_dir_path' : '/kaggle/working/MDB-stem-synth/audio_stems', 
    'mdb_hop_size' : 128/44100*3,
    'mdb_hop_size_dot' : 128*3,
    'mdb_out_class' : 360,
    'mdb_sr' : 44100,

    
    }

data_train_path = data_name+'_train_path'
data_validation_path = data_name+'_validation_path'
data_validation_csv_path = data_name+'_validation_csv_path'
data_test_path = data_name+'_test_path'
data_wav_dir_path = data_name+'_wav_dir_path'
data_hop_size = data_name+'_hop_size'
data_hop_size_dot = data_name +'_hop_size_dot'
data_out_class = data_name+'_out_class'
data_sr = data_name+'_sr'

train_path = data_dict[data_train_path]         # train_label_ark_path
validation_path = data_dict[data_validation_path]       # validation_label_ark_path
validation_csv_path = data_dict[data_validation_csv_path]       # ground_truth_path
test_csv_path = data_dict[data_test_path]       # test_label_ark_path
wav_dir_path = data_dict[data_wav_dir_path]       # wav_dir_path
hop_size = data_dict[data_hop_size]       # hop_size, such as 10 ms
hop_size_dot = data_dict[data_hop_size_dot]       # hop_size_dot, such as 10ms * 16000 (sr) = 160
out_class = data_dict[data_out_class]       # out_class
sr = data_dict[data_sr]       # Sampling Rate 
