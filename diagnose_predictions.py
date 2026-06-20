"""
Diagnostic: manually inspect predictions on individual dev files.

Instead of trusting the aggregate EER number, this prints per-file
predictions for a handful of bonafide and spoof dev examples so you can
see exactly what confidence the model assigns to each — useful for
sanity-checking whether the model is making real, calibrated judgments
or just outputting near-0/near-1 for everything (a sign of overconfident
memorization rather than genuine learning).

Run from your project root:
    python3 diagnose_predictions.py
"""

import os
import sys
import numpy as np
import librosa
import torch

sys.path.append(os.path.dirname(__file__))
from model import SpoofCNN

BASE = 'LA'
DEV_PROTOCOL = f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
DEV_AUDIO_DIR = f'{BASE}/ASVspoof2019_LA_dev/flac'
MODEL_CHECKPOINT = 'best_model.pt'

SAMPLE_RATE = 16000
DURATION = 4.0
N_MELS = 128
MAX_LEN = int(SAMPLE_RATE * DURATION)


def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    if len(audio) > MAX_LEN:
        start = (len(audio) - MAX_LEN) // 2
        audio = audio[start:start + MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)), mode='constant')
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=512, hop_length=160)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_norm


def load_protocol(path):
    entries = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label = 1 if parts[-1] == 'bonafide' else 0
            entries.append((file_id, label))
    return entries


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()

    entries = load_protocol(DEV_PROTOCOL)
    bonafide = [e for e in entries if e[1] == 1]
    spoof = [e for e in entries if e[1] == 0]

    np.random.seed(7)
    bona_sample = [bonafide[i] for i in np.random.choice(len(bonafide), 10, replace=False)]
    spoof_sample = [spoof[i] for i in np.random.choice(len(spoof), 10, replace=False)]

    print(f"{'File ID':<18} {'True Label':<12} {'P(bonafide)':<14} {'Correct?'}")
    print("-" * 60)

    all_scores_bona, all_scores_spoof = [], []

    with torch.no_grad():
        for file_id, label in bona_sample + spoof_sample:
            path = os.path.join(DEV_AUDIO_DIR, file_id + '.flac')
            mel = preprocess_audio(path)
            x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(x)
            score = torch.sigmoid(logit).item()

            true_str = 'bonafide' if label == 1 else 'spoof'
            pred_str = 'bonafide' if score > 0.5 else 'spoof'
            correct = '✓' if pred_str == true_str else '✗'

            print(f"{file_id:<18} {true_str:<12} {score:<14.6f} {correct}")

            if label == 1:
                all_scores_bona.append(score)
            else:
                all_scores_spoof.append(score)

    print(f"\nBonafide scores — mean: {np.mean(all_scores_bona):.4f}, "
          f"min: {np.min(all_scores_bona):.4f}, max: {np.max(all_scores_bona):.4f}")
    print(f"Spoof scores    — mean: {np.mean(all_scores_spoof):.4f}, "
          f"min: {np.min(all_scores_spoof):.4f}, max: {np.max(all_scores_spoof):.4f}")

    print(
        "\nIf scores cluster extremely close to exactly 0.0 or exactly 1.0 for "
        "every single file with almost no spread, that's consistent with "
        "memorization rather than learned, calibrated judgment."
    )


if __name__ == '__main__':
    main()