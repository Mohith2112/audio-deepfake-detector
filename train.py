"""
Training script for the spoof detection CNN.

Key fixes vs the original notebook:
1. WeightedRandomSampler — forces the model to see bonafide and spoof
   examples in roughly equal proportion each batch, instead of the raw
   ~9:1 imbalance. This is almost certainly why the original's val
   accuracy collapsed: with no correction, the easiest way to minimize
   training loss is to mostly predict "spoof".
2. Uses the OFFICIAL dev set for validation, not a slice of the train
   folder. Train/dev in ASVspoof19 are disjoint by speaker, so this
   validation number actually means something.
3. Early stopping on val EER (not val accuracy) — stops training the
   moment generalization starts to degrade, which is exactly the point
   where the original notebook's val_accuracy started falling apart.
4. Reports EER every epoch so you can see divergence happening live,
   instead of finding out only at the end.
5. MPS device support for Apple Silicon laptops.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.dirname(__file__))
from dataset import ASVspoofDataset
from model import SpoofCNN
from metrics import compute_eer


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_weighted_sampler(dataset):
    counts = dataset.get_class_counts()
    class_weight = {
        0: 1.0 / counts['spoof'],
        1: 1.0 / counts['bonafide'],
    }
    sample_weights = [class_weight[label] for label in dataset.labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for mels, labels in loader:
            mels, labels = mels.to(device), labels.float().to(device)
            logits = model(mels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * mels.size(0)

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    eer, _ = compute_eer(np.array(all_labels), np.array(all_scores))
    acc = np.mean((np.array(all_scores) > 0.5) == np.array(all_labels))

    return avg_loss, eer, acc


def train(
    train_protocol, train_audio_dir,
    dev_protocol, dev_audio_dir,
    epochs=20, batch_size=32, lr=1e-3,
    patience=4, checkpoint_path='best_model.pt'
):
    device = get_device()
    print(f"Using device: {device}")

    train_ds = ASVspoofDataset(train_protocol, train_audio_dir, augment=True)
    dev_ds = ASVspoofDataset(dev_protocol, dev_audio_dir, augment=False)

    print(f"Train: {len(train_ds)} files — {train_ds.get_class_counts()}")
    print(f"Dev:   {len(dev_ds)} files — {dev_ds.get_class_counts()}")

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SpoofCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_eer = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.float().to(device)

            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mels.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_eer, val_acc = evaluate(model, dev_loader, device, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_EER {val_eer*100:.2f}% | val_acc {val_acc*100:.2f}%"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best EER {best_eer*100:.2f}%, checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping — no EER improvement for {patience} epochs")
                break

    print(f"\nBest val EER: {best_eer*100:.2f}%")
    return model, best_eer


if __name__ == '__main__':
    BASE = 'LA'  # adjust to your actual path

    train(
        train_protocol=f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        train_audio_dir=f'{BASE}/ASVspoof2019_LA_train/flac',
        dev_protocol=f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        dev_audio_dir=f'{BASE}/ASVspoof2019_LA_dev/flac',
        epochs=20,
        batch_size=32,
        checkpoint_path='best_model.pt'
    )