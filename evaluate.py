"""
Evaluation + visualization script.

This replaces the original notebook's evaluation cells. Same plots
(confusion matrix, ROC, PR, calibration) but with two critical fixes:

1. Uses the corrected SpoofCNN (BatchNorm + global pooling) instead of
   the Flatten->48000-unit-Dense architecture that overfit.

2. IMPORTANT: this script works on any size eval set. Right now you only
   have 10 flac files (from test_eval.txt) — that's fine as a smoke test
   to confirm the pipeline runs end to end, but EER/AUC computed on 10
   samples is not a number you should report anywhere. Once you've
   downloaded the full ASVspoof2019_LA_eval set (71,237 files), point
   AUDIO_DIR and PROTOCOL_PATH at that instead and rerun this exact
   script for your real, reportable numbers.
"""

import os
import sys
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve

sys.path.append(os.path.dirname(__file__))
from model import SpoofCNN
from metrics import compute_eer

# ---- CONFIG: pointed at the FULL eval set (71,237 files) for real, reportable metrics ----
AUDIO_DIR = "LA/ASVspoof2019_LA_eval/flac"
PROTOCOL_PATH = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
MODEL_CHECKPOINT = "best_model.pt"       # saved from train.py
# -----------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION = 4.0
N_MELS = 128
MAX_LEN = int(SAMPLE_RATE * DURATION)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_protocol(path):
    """Parses speaker file_id - system_id label, robust to '-' system_id."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label_str = parts[-1]
            label = 1 if label_str == 'bonafide' else 0
            entries.append((file_id, label))
    return entries


def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)

    # Must match dataset.py's trimming exactly, or train/eval preprocessing
    # will be inconsistent and the model will see a different distribution
    # than it was trained on.
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > MAX_LEN:
        start = (len(audio) - MAX_LEN) // 2  # center crop for eval
        audio = audio[start:start + MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)), mode='constant')

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=512, hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def run_inference(model, device, entries, batch_size=64):
    """
    Batched inference — on 71k files, going one-at-a-time wastes a lot of
    time on Python/MPS call overhead. Batching gives a meaningful speedup.
    """
    from tqdm import tqdm

    y_true, y_scores, file_ids = [], [], []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(entries), batch_size), desc="Evaluating"):
            batch = entries[i:i + batch_size]
            mels, labels, ids = [], [], []

            for file_id, label in batch:
                path = os.path.join(AUDIO_DIR, file_id + '.flac')
                if not os.path.exists(path):
                    continue
                mel = preprocess_audio(path)
                mels.append(mel)
                labels.append(label)
                ids.append(file_id)

            if not mels:
                continue

            x = torch.tensor(np.stack(mels), dtype=torch.float32).unsqueeze(1).to(device)  # (B,1,n_mels,T)
            logits = model(x)
            scores = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(labels)
            y_scores.extend(scores)
            file_ids.extend(ids)

    return np.array(y_true), np.array(y_scores), file_ids


def main():
    device = get_device()
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"No checkpoint found at {MODEL_CHECKPOINT}. Run train.py first.")
        return

    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))

    entries = load_protocol(PROTOCOL_PATH)
    print(f"Evaluating on {len(entries)} files "
          f"({sum(1 for _, l in entries if l == 1)} bonafide, "
          f"{sum(1 for _, l in entries if l == 0)} spoof)")

    if len(entries) < 100:
        print(
            "\n*** NOTE: this is a small demo set. Treat results below as a "
            "pipeline smoke test only, NOT a reportable evaluation number. "
            "Rerun against the full ASVspoof2019_LA_eval set for real metrics. ***\n"
        )

    y_true, y_scores, file_ids = run_inference(model, device, entries)
    y_pred = (y_scores > 0.5).astype(int)

    # ---- EER (the metric that actually matters for this task) ----
    eer, eer_thresh = compute_eer(y_true, y_scores)
    print(f"EER: {eer*100:.2f}% (threshold {eer_thresh:.3f})")

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["spoof", "bonafide"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve'); plt.legend(loc="lower right")
    plt.savefig("roc_curve.png", bbox_inches='tight')
    plt.close()

    # ---- Precision-recall curve ----
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Avg precision = {avg_precision:.2f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-recall curve'); plt.legend(loc="lower left")
    plt.savefig("pr_curve.png", bbox_inches='tight')
    plt.close()

    # ---- Calibration curve (only meaningful with enough samples) ----
    if len(entries) >= 50:
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives')
        plt.title('Calibration curve'); plt.legend(loc="lower right")
        plt.savefig("calibration_curve.png", bbox_inches='tight')
        plt.close()
    else:
        print("Skipping calibration curve — needs >=50 samples to be meaningful "
              "(10-bin calibration on 10 files is not interpretable)")

    # ---- Class distribution ----
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_true)
    plt.xticks(ticks=[0, 1], labels=['spoof', 'bonafide'])
    plt.xlabel('Class'); plt.ylabel('Count'); plt.title('Class distribution')
    plt.savefig("class_distribution.png", bbox_inches='tight')
    plt.close()

    print("\nSaved: confusion_matrix.png, roc_curve.png, pr_curve.png, "
          "class_distribution.png" + (", calibration_curve.png" if len(entries) >= 50 else ""))


if __name__ == '__main__':
    main()