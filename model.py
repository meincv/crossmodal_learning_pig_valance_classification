"""
model.py — Two-stream multimodal ResNet-18 with late fusion
============================================================

Architecture overview
---------------------

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Modality 1: Audio Waveform  (1, SAMPLE_LENGTH)                     │
  │  ─────────────────────────────────────────────────────────────────  │
  │  1-D ResNet-18 → AdaptiveAvgPool → Flatten → 512-d feature vector   │
  └────────────────────────────────┬────────────────────────────────────┘
                                   │
                             Concatenation  ← 1024-d
                                   │
  ┌─────────────────────────────────┴───────────────────────────────────┐
  │  Modality 2: Spectrogram Image  (3, 224, 224)                       │
  │  ─────────────────────────────────────────────────────────────────  │
  │  2-D ResNet-18 → AdaptiveAvgPool → Flatten → 512-d feature vector   │
  └─────────────────────────────────────────────────────────────────────┘

                              ↓ concat (1024-d)
                     MLP: Linear → BN → ReLU → Dropout
                              ↓
                     Linear → 2 logits (Pos / Neg)

Key differences between the two streams
-----------------------------------------
Property            │ Stream 1 — Waveform         │ Stream 2 — Spectrogram
────────────────────┼──────────────────────────────┼───────────────────────
Input tensor        │ (1, SAMPLE_LENGTH)            │ (3, 224, 224)
Conv type           │ Conv1d (temporal)             │ Conv2d (spatial)
Input channels      │ 1 (mono signal)               │ 3 (RGB PNG)
Information         │ Raw amplitude over time       │ Frequency × time map
Pretrained weights  │ Random init (no ImageNet 1D)  │ ImageNet (torchvision)
Normalisation       │ Amplitude normalise to [-1,1] │ ImageNet mean/std

Why ResNet-18 for both?
-----------------------
ResNet-18 is a well-understood, moderately-sized backbone.
- The 2-D version is used as-is from torchvision, pretrained on ImageNet.
- The 1-D version is a minimal adaptation: all Conv2d → Conv1d, all
  BatchNorm2d → BatchNorm1d, all MaxPool2d → MaxPool1d, and the
  AdaptiveAvgPool collapses the time axis to 1.
  There are no pretrained 1-D ImageNet weights, so it is trained from
  scratch (random Kaiming initialisation).

Fusion strategy — Late fusion via concatenation
------------------------------------------------
Each branch independently extracts a 512-dimensional feature vector.
The two vectors are concatenated into a 1024-d vector which is then
passed through a two-layer MLP.

This is the simplest effective fusion strategy and directly maps to
the lecture concept of Representation Fusion (Slide 12/13).
More advanced options (cross-attention, bilinear pooling) are possible
but out of scope for this introductory lab.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# 1-D ResNet-18 building blocks
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock1d(nn.Module):
    """
    1-D version of the ResNet BasicBlock.
    Replaces every Conv2d / BN2d / MaxPool2d with their 1-D equivalents.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Shortcut: project if spatial size or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18_1D(nn.Module):
    """
    1-D ResNet-18 for raw audio waveforms.

    Input shape : (B, 1, SAMPLE_LENGTH)
    Output shape: (B, 512)  ← flattened feature vector

    Architecture mirrors the standard ResNet-18 layer count
    (8 BasicBlocks across 4 layer groups) but with 1-D convolutions.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # Entry stem: large kernel to capture broader temporal context
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 4 layer groups (2 BasicBlocks each = 8 blocks total → ResNet-18)
        self.layer1 = self._make_layer(64,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        # Global average pooling collapses the time axis to 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Kaiming initialisation — appropriate for ReLU activations
        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            BasicBlock1d(in_ch,  out_ch, stride=stride),
            BasicBlock1d(out_ch, out_ch, stride=1),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, SAMPLE_LENGTH)

        Returns
        -------
        features : (B, 512)
        """
        x = self.stem(x)      # (B, 64, T/4)
        x = self.layer1(x)    # (B, 64,  T/4)
        x = self.layer2(x)    # (B, 128, T/8)
        x = self.layer3(x)    # (B, 256, T/16)
        x = self.layer4(x)    # (B, 512, T/32)
        x = self.avgpool(x)   # (B, 512, 1)
        x = x.flatten(1)      # (B, 512)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2-D ResNet-18 for spectrograms
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18_2D(nn.Module):
    """
    2-D ResNet-18 for spectrogram images, pretrained on ImageNet.

    Input shape : (B, 3, 224, 224)
    Output shape: (B, 512)  ← flattened feature vector

    We use torchvision's pretrained ResNet-18 and remove its final
    classification head (the 1000-class fc layer), keeping only the
    feature extractor.  The AdaptiveAvgPool2d already collapses
    (H, W) → (1, 1), so after flattening we get a 512-d vector.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # Drop the classification head (avgpool is kept — it's part of backbone)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # backbone.children() order: conv1, bn1, relu, maxpool,
        #                            layer1, layer2, layer3, layer4, avgpool, fc
        # [:-1] removes fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        features : (B, 512)
        """
        x = self.features(x)   # (B, 512, 1, 1)
        x = x.flatten(1)       # (B, 512)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal fusion model
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalValenceClassifier(nn.Module):
    """
    Two-stream multimodal model for pig valence classification.

    Stream 1: 1-D ResNet-18 processes the raw audio waveform.
    Stream 2: 2-D ResNet-18 processes the spectrogram image.

    Both streams produce a 512-d feature vector.  These are
    concatenated into a 1024-d vector and passed through a small MLP
    that outputs class logits.

    Parameters
    ----------
    num_classes  : int  — number of output classes (2 for Pos/Neg)
    fusion_dim   : int  — feature size per branch (512 for ResNet-18)
    mlp_hidden   : int  — hidden layer size of the fusion MLP
    pretrained_2d: bool — use ImageNet weights for the 2-D branch
    """

    def __init__(
        self,
        num_classes: int  = Config.NUM_CLASSES,
        fusion_dim: int   = Config.FUSION_DIM,
        mlp_hidden: int   = Config.MLP_HIDDEN,
        pretrained_2d: bool = True,
    ):
        super().__init__()

        # ── Two independent feature extractors ────────────────────────────
        self.waveform_branch = ResNet18_1D(in_channels=1)
        self.spec_branch     = ResNet18_2D(pretrained=pretrained_2d)

        # ── MLP fusion head ───────────────────────────────────────────────
        # Input: concatenated 512 + 512 = 1024-d vector
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim * 2, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(mlp_hidden, num_classes),
        )

        # Initialise only the MLP head (branches have their own init)
        self._init_head()

    def _init_head(self):
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        waveform: torch.Tensor,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform    : (B, 1, SAMPLE_LENGTH)
        spectrogram : (B, 3, 224, 224)

        Returns
        -------
        logits : (B, num_classes)  — raw pre-softmax scores
        """
        # ── Extract features from each modality independently ─────────────
        feat_wave = self.waveform_branch(waveform)      # (B, 512)
        feat_spec = self.spec_branch(spectrogram)       # (B, 512)

        # ── Concatenate (late/feature fusion) ─────────────────────────────
        # This is the simplest multimodal fusion strategy.
        # Lecture connection: "Fusion" in Representation Learning (Slide 12).
        fused = torch.cat([feat_wave, feat_spec], dim=1)  # (B, 1024)

        # ── Classify ──────────────────────────────────────────────────────
        logits = self.fusion_head(fused)                # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Architecture sanity check (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = MultimodalValenceClassifier(
        num_classes=Config.NUM_CLASSES,
        fusion_dim=Config.FUSION_DIM,
        mlp_hidden=Config.MLP_HIDDEN,
    )

    total = sum(p.numel() for p in model.parameters())
    wave_params = sum(p.numel() for p in model.waveform_branch.parameters())
    spec_params = sum(p.numel() for p in model.spec_branch.parameters())
    head_params = sum(p.numel() for p in model.fusion_head.parameters())

    print("MultimodalValenceClassifier — parameter counts:")
    print(f"  Waveform branch (1D ResNet-18) : {wave_params:>10,}")
    print(f"  Spectrogram branch (2D ResNet-18): {spec_params:>8,}")
    print(f"  Fusion MLP head                : {head_params:>10,}")
    print(f"  Total                          : {total:>10,}")

    # Forward-pass sanity check with random tensors
    B = 4
    dummy_wave = torch.randn(B, 1, Config.SAMPLE_LENGTH)
    dummy_spec = torch.randn(B, 3, Config.SPEC_SIZE, Config.SPEC_SIZE)

    with torch.no_grad():
        feat_w = model.waveform_branch(dummy_wave)
        feat_s = model.spec_branch(dummy_spec)
        logits = model(dummy_wave, dummy_spec)

    print(f"\nForward-pass shapes (batch size = {B}):")
    print(f"  Waveform input       : {tuple(dummy_wave.shape)}")
    print(f"  Waveform features    : {tuple(feat_w.shape)}")
    print(f"  Spectrogram input    : {tuple(dummy_spec.shape)}")
    print(f"  Spectrogram features : {tuple(feat_s.shape)}")
    print(f"  Logits               : {tuple(logits.shape)}")
