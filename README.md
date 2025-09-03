# YOLOPitch: A Time-Frequency Dual-Branch Neural Network for Pitch Estimation

YOLOPitch Architecture

*Current Implementation Status: Class Imbalance Challenge in PTDB Dataset*

## Overview

YOLOPitch is a time-frequency dual-branch neural network model for pitch estimation that mimics human pitch perception mechanisms. The model simultaneously processes both waveform (temporal information) and STFT spectrum (spectral information) to achieve state-of-the-art pitch estimation performance.

## Current Implementation Challenge

⚠️ **Class Imbalance Issue in PTDB Dataset** ⚠️

The current implementation shows a concerning behavior pattern during training:

- **High Overall Accuracy** (>95%)
- **0% Voice Accuracy** (never detects voiced frames)
- **0% RPA/RCA** (no pitch estimation capability)

This indicates the model may be learning to always predict "silence" (class 0) due to the extreme class imbalance present in the PTDB dataset, where approximately 80-90% of frames contain silence.

## Important Note on Potential Solutions

While the class imbalance issue is clearly present, **the optimal solution methodology remains uncertain**. The original paper does not explicitly detail how the authors addressed this challenge in their implementation, despite reporting excellent results (92.7% F-Score, 90.4% RPA on PTDB).

Various approaches could potentially address this issue, but without further verification against the original implementation or additional guidance from the authors, **no single solution can be confidently recommended** as the correct approach used in the paper.

## Community Support Requested

Given the uncertainty around the proper implementation strategy for handling class imbalance in YOLOPitch:

- Researchers who have successfully implemented YOLOPitch are encouraged to share their approaches
- The original authors may wish to clarify their methodology for handling class imbalance
- The community is invited to contribute insights on effective strategies for pitch estimation with imbalanced datasets

## References

Xufei Li, Hao Huang, Ying Hu, Liang He, Jiabao Zhang, Yuyi Wang. [YOLOPitch: A Time-Frequency Dual-Branch YOLO Model for Pitch Estimation](https://www.isca-archive.org/interspeech_2024/li24ja_interspeech.pdf). INTERSPEECH 2024.

License 

This implementation is licensed under the MIT License. 