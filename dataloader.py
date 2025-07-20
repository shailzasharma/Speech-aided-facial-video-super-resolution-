#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import argparse
import random
from glob import glob
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

# ---------------------------------------------------------------------------
# Audio hyperparameters 
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_N_FFT: int = 1024
AUDIO_HOP_LENGTH: int = 200        # (≈ 12.5 ms at 16 kHz) adjust to match preprocessing
AUDIO_WIN_LENGTH: int = 800        # (≈ 50 ms)
AUDIO_N_MELS: int = 80
AUDIO_FMIN: int = 55
AUDIO_FMAX: int = 7600             # Must be <= sample_rate // 2

# Derived constant used in temporal alignment; 80 mel time steps per second assumption
# If the preprocessing differs, update MEL_STEPS_PER_VIDEO_FRAME accordingly.
VIDEO_FPS: float = 25.0
MEL_STEPS_PER_SECOND: float = AUDIO_SAMPLE_RATE / AUDIO_HOP_LENGTH  # ≈ 80.0
MEL_STEPS_PER_VIDEO_FRAME: float = MEL_STEPS_PER_SECOND / VIDEO_FPS  # ≈ 3.2

# ---------------------------------------------------------------------------
# Temporal / spatial window constants 
# ---------------------------------------------------------------------------
WIN_LEN: int = 5            # Number of consecutive frames per sample (temporal depth)
MEL_WIN_SIZE: int = 16      # Number of mel frames per aligned audio chunk
FRAME_SIZE: int = 128       # Target square size for high‑res frames
FRAME_SIZE_64: int = 64
FRAME_SIZE_32: int = 32

# ---------------------------------------------------------------------------
# audio feature helpers 
# ---------------------------------------------------------------------------

def load_wav(path: str, sr: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Load waveform as float32 mono at target sample rate."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav.astype(np.float32)


def _mel_filter_bank() -> np.ndarray:
    return librosa.filters.mel(
        sr=AUDIO_SAMPLE_RATE,
        n_fft=AUDIO_N_FFT,
        n_mels=AUDIO_N_MELS,
        fmin=AUDIO_FMIN,
        fmax=AUDIO_FMAX
    )

_MEL_BASIS: np.ndarray | None = None


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """Compute log‑mel spectrogram (dB) without external hparams module.

    Returns shape (n_mel_frames, N_MELS).
    """
    global _MEL_BASIS
    if _MEL_BASIS is None:
        _MEL_BASIS = _mel_filter_bank()
    stft = librosa.stft(
        wav,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP_LENGTH,
        win_length=AUDIO_WIN_LENGTH,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    magnitude = np.abs(stft)
    mel_spec = np.dot(_MEL_BASIS, magnitude)
    # Numerical stability floor
    mel_spec = np.maximum(1e-10, mel_spec)
    log_mel = 20.0 * np.log10(mel_spec)
    # Simple normalization to roughly [-1,1]
    log_mel_norm = (log_mel + 100.0) / 100.0  # shift & scale (dataset dependent)
    return log_mel_norm.T  # (time, n_mels)

# ---------------------------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------------------------

def discover_video_dirs(data_root: str, split: str) -> List[str]:
    split_root = os.path.join(data_root, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split directory not found: {split_root}")
    video_dirs: List[str] = []
    for root, _dirs, files in os.walk(split_root):
        if 'audio.wav' in files and any(f.endswith('.jpg') for f in files):
            video_dirs.append(root)
    if not video_dirs:
        raise RuntimeError(f"No frame+audio directories under {split_root}")
    return sorted(video_dirs)

# ---------------------------------------------------------------------------
# Frame & mel helpers
# ---------------------------------------------------------------------------

def frame_id(frame_path: str) -> int:
    return int(os.path.splitext(os.path.basename(frame_path))[0])


def build_frame_window(start_frame_path: str, length: int) -> Optional[List[str]]:
    start = frame_id(start_frame_path)
    vid_dir = os.path.dirname(start_frame_path)
    paths: List[str] = []
    for i in range(start, start + length):
        fp = os.path.join(vid_dir, f"{i}.jpg")
        if not os.path.isfile(fp):
            return None
        paths.append(fp)
    return paths


def read_and_resize_frames(paths: Sequence[str], target: int) -> Optional[List[np.ndarray]]:
    frames: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            img = cv2.resize(img, (target, target))
        except Exception:
            return None
        frames.append(img)
    return frames


def mel_chunk(spec: np.ndarray, one_based_frame_index: int) -> np.ndarray:
    """Return MEL_WIN_SIZE mel frames aligned to a (1‑based) video frame index."""
    start = int(MEL_STEPS_PER_VIDEO_FRAME * one_based_frame_index)
    end = start + MEL_WIN_SIZE
    return spec[start:end, :]


def segmented_mel_sequence(spec: np.ndarray, center_frame_path: str) -> Optional[np.ndarray]:
    """Build sequence of mel chunks for temporal window.

    Original logic (shifted naming): center = frame_id + 1, iterate [center, center+WIN_LEN)
    each requesting mel aligned to (i - 2).
    """
    center = frame_id(center_frame_path) + 1
    if center - 2 < 0:
        return None
    segments = []
    for i in range(center, center + WIN_LEN):
        seg = mel_chunk(spec, i - 2)
        if seg.shape[0] != MEL_WIN_SIZE:
            return None
        segments.append(seg.T)  # (N_MELS, MEL_WIN_SIZE)
    return np.asarray(segments)


def normalize_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(frames) / 255.0  # (T,H,W,C)
    return arr

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AVWindowDataset(torch.utils.data.Dataset):
    """Randomly samples short audio‑visual windows with multi‑scale frames + mel features."""
    def __init__(self, data_root: str, split: str, seed: int | None = None):
        self.data_root = data_root
        self.split = split
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.video_dirs = discover_video_dirs(data_root, split)

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, index: int):  # index ignored; random sampling each call
        while True:
            vid_dir = random.choice(self.video_dirs)
            frame_paths = glob(os.path.join(vid_dir, '*.jpg'))
            if len(frame_paths) <= 2 * WIN_LEN:
                continue
            anchor = random.choice(frame_paths)
            window_files = build_frame_window(anchor, WIN_LEN)
            if window_files is None:
                continue
            frames = read_and_resize_frames(window_files, FRAME_SIZE)
            if frames is None:
                continue
            wav_path = os.path.join(vid_dir, 'audio.wav')
            if not os.path.isfile(wav_path):
                continue
            try:
                wav = load_wav(wav_path)
                full_mel = melspectrogram(wav)  # (time, N_MELS)
            except Exception:
                continue
            center_mel_arr = mel_chunk(full_mel.copy(), frame_id(anchor))
            if center_mel_arr.shape[0] != MEL_WIN_SIZE:
                continue
            mel_seq = segmented_mel_sequence(full_mel.copy(), anchor)
            if mel_seq is None:
                continue
            # Frames: (T,H,W,C) -> (C,T,H,W)
            vid_np = normalize_frames(frames)
            vid_np = np.transpose(vid_np, (3,0,1,2))
            frames_128 = torch.FloatTensor(vid_np)
            frames_64  = F.interpolate(frames_128, size=[FRAME_SIZE_64, FRAME_SIZE_64], mode='bilinear', align_corners=True)
            frames_32  = F.interpolate(frames_64,  size=[FRAME_SIZE_32, FRAME_SIZE_32], mode='bilinear', align_corners=True)
            center_mel = torch.FloatTensor(center_mel_arr.T).unsqueeze(0)        # (1,N_MELS,MEL_WIN_SIZE)
            segmented_mels = torch.FloatTensor(mel_seq).unsqueeze(1)             # (T,1,N_MELS,MEL_WIN_SIZE)
            return frames_32, segmented_mels, center_mel, frames_128, frames_64


