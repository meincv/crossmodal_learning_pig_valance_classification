"""
config.py — Centralised hyperparameters and paths
==================================================
All settings live here. Every other module imports from this file.
Change a value once and the whole project picks it up automatically.
"""


class Config:
    # ── Audio ──────────────────────────────────────────────────────────────
    # Target sample rate used when loading .wav files.
    # The actual file sample rate is read at load time (see dataset.py).
    # If the file SR differs from this value, librosa resamples automatically.
    SAMPLE_RATE   = 44_100        # Hz
    # Fixed number of raw audio samples per clip fed to the waveform branch.
    # 44 100 samples ≈ 1 second.  Must be a power of 2 for the 1-D ResNet
    # downsampling to work cleanly (we use adaptive pooling, so any length
    # works, but a round number makes shape reasoning easier).
    SAMPLE_LENGTH = 44_100

    # ── Spectrogram ────────────────────────────────────────────────────────
    # Native spectrogram image size (875 × 656 pixels, centred content).
    # Both dimensions are resized to SPEC_SIZE before feeding to ResNet-18.
    SPEC_IMG_W    = 875
    SPEC_IMG_H    = 656
    SPEC_SIZE     = 224           # ResNet-18 canonical input size

    # ── Paths ──────────────────────────────────────────────────────────────
    ANNOTATIONS_FILE  = "annotations.xlsx"
    AUDIO_DIR         = "audio/"
    SPEC_DIR          = "spectrograms/"
    BEST_MODEL_PATH   = "best_model.pth"

    # ── Column names (must match the annotation spreadsheet exactly) ───────
    AUDIO_COL   = "Audio Filename"
    SPEC_COL    = "Spectrogram Filename"
    LABEL_COL   = "valence"               # raw string: "Pos" or "Neg"

    # ── Model ──────────────────────────────────────────────────────────────
    NUM_CLASSES    = 2     # binary: Positive (1) vs Negative (0)
    FUSION_DIM     = 512   # ResNet-18 output channels per branch → concat = 1024
    MLP_HIDDEN     = 256   # hidden size of the MLP fusion head

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE    = 16
    NUM_EPOCHS    = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    PATIENCE      = 5      # epochs without val-loss improvement before LR drop
    NUM_WORKERS   = 4      # DataLoader workers (set 0 on Windows)

    # ── Reproducibility ────────────────────────────────────────────────────
    SEED          = 42

    # ── Split — grouped by Recording Team ──────────────────────────────────
    # SoundWel recordings come from different farms / recording sessions.
    # We group by team to prevent data from the same recording session
    # appearing in both train and test (avoids within-session leakage).
    # If your annotation file uses a different column name, update here.
    GROUP_COLUMN  = "Recording Team"
