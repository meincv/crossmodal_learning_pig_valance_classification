"""
train.py — Training, validation and test evaluation
====================================================
Run with:
    python train.py

What this script does
---------------------
1.  Loads the annotation spreadsheet and builds group-aware splits.
2.  Creates DataLoaders for train / val / test.
3.  Instantiates MultimodalValenceClassifier and a weighted cross-entropy
    loss (to handle class imbalance).
4.  Trains for up to NUM_EPOCHS epochs, saving the best checkpoint
    based on validation loss.
5.  After training, loads the best checkpoint and evaluates on the
    held-out test set.
6.  Saves loss/accuracy curves and a confusion matrix to PNG files.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config   import Config
from model    import MultimodalValenceClassifier
from dataset  import SoundWelDataset, build_splits
from evaluate import evaluate, print_test_report

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────────────────────────────────
df_train, df_val, df_test = build_splits(Config.ANNOTATIONS_FILE)

train_ds = SoundWelDataset(df_train, augment=True)
val_ds   = SoundWelDataset(df_val,   augment=False)
test_ds  = SoundWelDataset(df_test,  augment=False)

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                          shuffle=True,  num_workers=Config.NUM_WORKERS,
                          pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                          shuffle=False, num_workers=Config.NUM_WORKERS,
                          pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE,
                          shuffle=False, num_workers=Config.NUM_WORKERS,
                          pin_memory=True)

print(f"Batches per epoch — train: {len(train_loader)} | "
      f"val: {len(val_loader)} | test: {len(test_loader)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────
model = MultimodalValenceClassifier(
    num_classes=Config.NUM_CLASSES,
    fusion_dim=Config.FUSION_DIM,
    mlp_hidden=Config.MLP_HIDDEN,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"MultimodalValenceClassifier — {total_params:,} parameters\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss, optimiser, scheduler
# ─────────────────────────────────────────────────────────────────────────────
# Weighted cross-entropy to handle class imbalance.
neg_count = (df_train["label"] == 0).sum()
pos_count = (df_train["label"] == 1).sum()
class_weights = torch.tensor(
    [pos_count / neg_count, 1.0], dtype=torch.float
).to(DEVICE)
print(f"Class weights → Neg: {class_weights[0]:.3f}  Pos: {class_weights[1]:.3f}\n")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Adam is a sensible default for multimodal models — easier to tune than SGD.
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
)

# Automatically reduce LR when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.2, patience=Config.PATIENCE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    One full training pass.  Returns (avg_loss, accuracy).
    NOTE: the DataLoader now yields (waveform, spectrogram, label) triples.
    """
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for waveforms, spectrograms, labels in loader:
        waveforms    = waveforms.to(device)
        spectrograms = spectrograms.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms, spectrograms)    # ← two inputs
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(labels)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    from sklearn.metrics import accuracy_score
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")

print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
print("─" * 65)

for epoch in range(1, Config.NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    val_loss, val_acc, _, _ = evaluate(
        model, val_loader, criterion, DEVICE
    )
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]
    flag = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        flag = "  ← saved"

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
          f"{val_loss:>8.4f}  {val_acc:>7.4f}  {current_lr:>8.6f}{flag}")

print("\nTraining complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Test evaluation
# ─────────────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=DEVICE))
test_loss, test_acc, test_labels, test_probs = evaluate(
    model, test_loader, criterion, DEVICE
)
print()
auc, ap = print_test_report(test_labels, test_probs, test_loss, test_acc)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────────────────────────────────────
epochs_range = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["val_loss"],   label="Val")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss curves"); axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train")
axes[1].plot(epochs_range, history["val_acc"],   label="Val")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy curves"); axes[1].legend()

cm = confusion_matrix(test_labels, (test_probs >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"], ax=axes[2])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title(f"Confusion matrix — AUC={auc:.3f}")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("Plot saved → training_results.png")
