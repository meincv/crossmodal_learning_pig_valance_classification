"""
dataset.py — Data loading and train/val/test splitting
=======================================================

Two-modality pipeline
---------------------
This dataset loads TWO modalities per sample (cross-modal learning):

  Modality 1 — Audio waveform (.wav)
    • Raw 1-D signal loaded with librosa (resamples if needed).
    • Padded or truncated to exactly SAMPLE_LENGTH samples.
    • Shape returned: (1, SAMPLE_LENGTH)  ← (channels, time)

  Modality 2 — Spectrogram image (.png)
    • Pre-computed spectrogram stored as a 875×656 PNG file.
    • Resized to 224×224 and normalised with ImageNet statistics
      (because ResNet-18 was pretrained on ImageNet images).
    • Shape returned: (3, 224, 224)  ← (RGB channels, H, W)

Why two modalities?
-------------------
The waveform captures the raw temporal dynamics of the vocalisation —
amplitude, rhythm, fine timing.  The spectrogram makes frequency content
explicit and visually structured.  Fusing them in a single model lets the
network exploit both complementary sources of information (see lecture:
Interconnected Modalities, Heterogeneity dimensions).

Split strategy — grouped by Recording Team
------------------------------------------
Identical rationale to the pig_valence_classification project:
grouping by Recording Team prevents data from the same recording
session appearing in both train and test sets.
If the SoundWel annotations do not include a "Recording Team" column,
the split falls back to a random 70/15/15 split with a fixed seed.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    import soundfile as sf
    LIBROSA_AVAILABLE = False

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Waveform helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = Config.SAMPLE_RATE) -> np.ndarray:
    """
    Load a .wav file and return a 1-D float32 array at *target_sr* Hz.

    The sample rate is read from the file header at load time.
    librosa resamples automatically if the file SR differs from target_sr.
    soundfile is used as a fast fallback (no resampling).
    """
    if LIBROSA_AVAILABLE:
        y, _ = librosa.load(path, sr=target_sr, mono=True)
    else:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)       # stereo → mono
        if sr != target_sr:
            raise ValueError(
                f"{path}: sample rate {sr} ≠ {target_sr}. "
                "Install librosa for automatic resampling."
            )
    return y.astype(np.float32)


def pad_or_truncate(y: np.ndarray, length: int) -> np.ndarray:
    """Force a 1-D array to exactly *length* samples."""
    if len(y) >= length:
        return y[:length]
    return np.pad(y, (0, length - len(y)), mode="constant")


# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram transform
# ─────────────────────────────────────────────────────────────────────────────

def build_spec_transform(augment: bool = False) -> transforms.Compose:
    """
    Build a torchvision transform pipeline for spectrogram images.

    The spectrograms are 875×656 PNG files with the content centred.
    We resize to 224×224 (ResNet-18 canonical input) and normalise with
    the ImageNet mean/std because the visual branch is initialised with
    ImageNet weights.

    Augmentation (training only):
      • Random horizontal flip  — time-reversal of the spectrogram
      • Random colour jitter    — simulate recording condition variation
    """
    base = [
        transforms.Resize((Config.SPEC_SIZE, Config.SPEC_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet
                             std =[0.229, 0.224, 0.225]),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SoundWelDataset(Dataset):
    """
    Loads pig vocalisations from SoundWel as a pair of modalities:
      (waveform_tensor, spectrogram_tensor, label)

    Parameters
    ----------
    dataframe     : pd.DataFrame — rows from the annotation spreadsheet
    audio_dir     : str          — folder containing .wav files
    spec_dir      : str          — folder containing .png spectrogram files
    sample_length : int          — fixed number of waveform samples per clip
    augment       : bool         — apply training-time augmentation
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        audio_dir: str = Config.AUDIO_DIR,
        spec_dir: str  = Config.SPEC_DIR,
        sample_length: int = Config.SAMPLE_LENGTH,
        augment: bool = False,
    ):
        self.df            = dataframe.reset_index(drop=True)
        self.audio_dir     = audio_dir
        self.spec_dir      = spec_dir
        self.sample_length = sample_length
        self.spec_tf       = build_spec_transform(augment=augment)
        self.augment       = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Modality 1: Audio waveform ─────────────────────────────────────
        audio_path = os.path.join(self.audio_dir, row[Config.AUDIO_COL])
        y = load_audio(audio_path, target_sr=Config.SAMPLE_RATE)
        y = pad_or_truncate(y, self.sample_length)

        if self.augment:
            y = self._augment_waveform(y)

        # Amplitude normalisation to [-1, 1]
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

        # (1, SAMPLE_LENGTH) — same channel convention as the pig project
        waveform = torch.from_numpy(y).unsqueeze(0)

        # ── Modality 2: Spectrogram image ──────────────────────────────────
        spec_path = os.path.join(self.spec_dir, row[Config.SPEC_COL])
        img = Image.open(spec_path).convert("RGB")   # ensure 3-channel PNG
        spectrogram = self.spec_tf(img)               # (3, 224, 224)

        # ── Label ──────────────────────────────────────────────────────────
        label = torch.tensor(row["label"], dtype=torch.long)

        return waveform, spectrogram, label

    # ── Waveform augmentation ─────────────────────────────────────────────
    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        """
        Lightweight waveform augmentations for training.
        Same strategy as pig_valence_classification:

        1. Random gain   — simulate microphone distance variation.
        2. Random flip   — time reversal as regularisation.
        3. Additive noise — background noise simulation.
        """
        # 1. Random gain [0.8, 1.2]
        y = y * np.random.uniform(0.8, 1.2)
        # 2. Time reversal (~50 % chance)
        if np.random.rand() < 0.5:
            y = y[::-1].copy()
        # 3. Low-level Gaussian noise (SNR ≈ 30 dB)
        noise_std = np.abs(y).mean() * 0.03
        y = y + np.random.randn(len(y)).astype(np.float32) * noise_std
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Split builder
# ─────────────────────────────────────────────────────────────────────────────

def build_splits(annotations_file: str = Config.ANNOTATIONS_FILE):
    """
    Load the annotation spreadsheet and return three DataFrames:
    (df_train, df_val, df_test).

    Group-aware splitting (same logic as pig_valence_classification):
      • Rows are grouped by Config.GROUP_COLUMN (e.g. "Recording Team").
      • No group appears in more than one split, preventing within-session
        data leakage.
      • If the GROUP_COLUMN is missing, falls back to random splitting.

    Returns
    -------
    df_train, df_val, df_test : pd.DataFrame
    """
    df = pd.read_excel(annotations_file)

    # Binary label: "Pos" → 1, "Neg" → 0
    df["label"] = (df[Config.LABEL_COL].str.strip() == "Pos").astype(int)

    # ── Group-aware split ──────────────────────────────────────────────────
    if Config.GROUP_COLUMN in df.columns:
        groups = df[Config.GROUP_COLUMN].values

        # Step 1: hold out 20 % as (val + test)
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.20,
                                      random_state=Config.SEED)
        train_idx, temp_idx = next(gss_outer.split(df, groups=groups))

        # Step 2: split the 20 % evenly into val and test
        temp_groups = groups[temp_idx]
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50,
                                      random_state=Config.SEED)
        val_rel, test_rel = next(
            gss_inner.split(df.iloc[temp_idx], groups=temp_groups)
        )
        val_idx  = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

    else:
        # Fallback: random 70 / 15 / 15 split
        print(f"[Warning] Column '{Config.GROUP_COLUMN}' not found — "
              "using random split (no group leakage protection).")
        rng     = np.random.default_rng(Config.SEED)
        idx     = rng.permutation(len(df))
        n_train = int(0.70 * len(df))
        n_val   = int(0.15 * len(df))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train: n_train + n_val]
        test_idx  = idx[n_train + n_val:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    print("Split summary:")
    for name, dset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        pos = dset["label"].sum()
        neg = len(dset) - pos
        print(f"  {name:<5}  {len(dset):>5} samples  |  Pos: {pos}  Neg: {neg}")

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_train, df_val, df_test = build_splits()

    train_ds = SoundWelDataset(df_train, augment=True)
    val_ds   = SoundWelDataset(df_val,   augment=False)
    test_ds  = SoundWelDataset(df_test,  augment=False)

    print(f"\nDataset sizes — train: {len(train_ds)} | "
          f"val: {len(val_ds)} | test: {len(test_ds)}")

    waveform, spectrogram, label = train_ds[0]
    print(f"\nFirst training sample:")
    print(f"  Waveform tensor shape   : {tuple(waveform.shape)}")
    print(f"  Spectrogram tensor shape: {tuple(spectrogram.shape)}")
    print(f"  Label                   : {label.item()}  (0=Neg, 1=Pos)")
