# SoundWel Multimodal Valence Classifier

A two-stream multimodal deep learning project for binary pig valence classification
(Positive / Negative) using the **SoundWel** dataset.

This project is part of the **Pattern Learning Fundamentals (PLF)** course lab on
*Cross-Modal and Multimodal Learning*.

---

## Overview

Each pig vocalisation in SoundWel is represented by **two modalities**:

| Modality | Format | Shape fed to model | Backbone |
|----------|--------|--------------------|----------|
| Audio waveform | `.wav` | `(1, 44100)` | 1-D ResNet-18 (trained from scratch) |
| Spectrogram image | `.png` (875×656) | `(3, 224, 224)` | 2-D ResNet-18 (ImageNet pretrained) |

Both branches produce a **512-d feature vector**.  These are **concatenated** into a
1024-d vector and passed through a small MLP that outputs Pos / Neg logits.

```
Waveform  (1, L) ──► 1D ResNet-18 ──► 512-d ──┐
                                                ├──► concat 1024-d ──► MLP ──► ŷ
Spectrogram (3,H,W) ► 2D ResNet-18 ──► 512-d ──┘
```

---

## Project structure

```
soundwel_multimodal/
├── config.py          ← All hyperparameters and file paths
├── dataset.py         ← SoundWelDataset + build_splits()
├── model.py           ← MultimodalValenceClassifier
├── evaluate.py        ← evaluate() + print_test_report()
├── train.py           ← Training loop, scheduler, plots
├── soundwel_demo.ipynb← Interactive walkthrough notebook
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install torch torchvision torchaudio librosa \
            scikit-learn pandas openpyxl Pillow seaborn matplotlib
```

### 2. Organise data

```
soundwel_multimodal/
├── annotations.xlsx       ← columns: Audio Filename, Spectrogram Filename, valence
├── audio/                 ← .wav files
│   └── *.wav
└── spectrograms/          ← .png files (875×656 px)
    └── *.png
```

### 3. Verify paths

Open `config.py` and confirm `AUDIO_DIR`, `SPEC_DIR`, `ANNOTATIONS_FILE`,
`AUDIO_COL`, `SPEC_COL`, and `LABEL_COL` match your data.

### 4. Run in order

```bash
python dataset.py   # verify splits and tensor shapes
python model.py     # verify architecture and forward-pass shapes
python train.py     # train, validate, test, save plots
```

Or open `soundwel_demo.ipynb` for an interactive walkthrough.

---

## Key differences between the two ResNet-18 streams

| Property | Waveform stream | Spectrogram stream |
|---|---|---|
| Input tensor | `(1, sample_length)` — 1-D signal | `(3, 224, 224)` — 2-D image |
| Conv type | `Conv1d` — slides along time axis | `Conv2d` — slides over H × W |
| Information captured | Temporal amplitude dynamics | Frequency × time structure |
| Pretrained weights | Random Kaiming init (no 1-D ImageNet) | ImageNet IMAGENET1K_V1 |
| Normalisation | Peak amplitude norm to `[−1, 1]` | ImageNet mean/std |
| Output | 512-d feature vector | 512-d feature vector |

---

## Expected results

Results will vary by dataset size and split.  Typical ranges observed:

| Metric | Waveform only | Spectrogram only | Both (multimodal) |
|--------|--------------|-----------------|-------------------|
| Accuracy | ~65–70 % | ~68–73 % | ~72–78 % |
| AUC-ROC | ~0.68–0.74 | ~0.70–0.76 | ~0.74–0.82 |

---


