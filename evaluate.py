"""
evaluate.py — Validation / test evaluation helpers
====================================================
Keeps metric computation cleanly separated from the training loop.
Import and call these functions from train.py.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from config import Config


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Run one full pass over *loader* without gradient computation.

    Returns
    -------
    avg_loss  : float
    accuracy  : float
    all_labels: list[int]
    all_probs : list[float]  ← predicted probability for class 1 (Pos)
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for waveforms, spectrograms, labels in loader:
            waveforms   = waveforms.to(device)
            spectrograms = spectrograms.to(device)
            labels      = labels.to(device)

            logits = model(waveforms, spectrograms)
            loss   = criterion(logits, labels)

            probs  = torch.softmax(logits, dim=1)[:, 1]   # P(Pos)
            preds  = logits.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, np.array(all_labels), np.array(all_probs)


def print_test_report(
    labels: np.ndarray,
    probs: np.ndarray,
    loss: float,
    accuracy: float,
) -> tuple:
    """
    Print a full test-set evaluation report and return (AUC, AP).
    """
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)

    print("=" * 50)
    print("Test-set evaluation")
    print("=" * 50)
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")
    print()
    print(classification_report(labels, preds,
                                 target_names=["Negative", "Positive"]))
    return auc, ap
