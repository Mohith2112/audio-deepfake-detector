"""
Diagnostic: check for non-content artifacts (silence padding, amplitude,
leading/trailing silence) that could let the model cheat without ever
learning real spoofing cues.

This is the next most likely explanation after ruling out file leakage
and duration. ASVspoof bonafide and spoof files are sourced/processed
slightly differently upstream (different TTS/VC engines, different
silence trimming), and a CNN on raw mel spectrograms can sometimes
latch onto those processing artifacts rather than voice content.

Run from your project root:
    python3 diagnose_silence.py
"""

import os
import numpy as np
import librosa

BASE = 'LA'
DEV_PROTOCOL = f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
DEV_AUDIO_DIR = f'{BASE}/ASVspoof2019_LA_dev/flac'

SAMPLE_RATE = 16000
N_SAMPLE = 150


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


def silence_stats(audio, top_db=30):
    """Returns leading silence (s), trailing silence (s), and % of audio that's silence."""
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return len(audio) / SAMPLE_RATE, len(audio) / SAMPLE_RATE, 1.0

    leading = intervals[0][0] / SAMPLE_RATE
    trailing = (len(audio) - intervals[-1][1]) / SAMPLE_RATE
    voiced_samples = sum(end - start for start, end in intervals)
    silence_pct = 1 - (voiced_samples / len(audio))

    return leading, trailing, silence_pct


def main():
    entries = load_protocol(DEV_PROTOCOL)
    bonafide = [e for e in entries if e[1] == 1]
    spoof = [e for e in entries if e[1] == 0]

    np.random.seed(3)
    n_b = min(len(bonafide), N_SAMPLE)
    n_s = min(len(spoof), N_SAMPLE)
    bona_sample = [bonafide[i] for i in np.random.choice(len(bonafide), n_b, replace=False)]
    spoof_sample = [spoof[i] for i in np.random.choice(len(spoof), n_s, replace=False)]

    def collect_stats(file_list):
        leadings, trailings, silences, rms_means, max_amps = [], [], [], [], []
        for file_id, _ in file_list:
            path = os.path.join(DEV_AUDIO_DIR, file_id + '.flac')
            if not os.path.exists(path):
                continue
            audio, _ = librosa.load(path, sr=SAMPLE_RATE)
            lead, trail, sil = silence_stats(audio)
            leadings.append(lead)
            trailings.append(trail)
            silences.append(sil)
            rms_means.append(np.sqrt(np.mean(audio**2)))
            max_amps.append(np.max(np.abs(audio)))
        return {
            'leading_silence': np.array(leadings),
            'trailing_silence': np.array(trailings),
            'silence_pct': np.array(silences),
            'rms': np.array(rms_means),
            'max_amp': np.array(max_amps),
        }

    print("Computing stats for bonafide files...")
    bona_stats = collect_stats(bona_sample)
    print("Computing stats for spoof files...")
    spoof_stats = collect_stats(spoof_sample)

    print("\n=== Comparison: bonafide vs spoof ===\n")
    for key in bona_stats:
        b = bona_stats[key]
        s = spoof_stats[key]
        print(f"{key}:")
        print(f"  bonafide  mean={b.mean():.4f}  std={b.std():.4f}")
        print(f"  spoof     mean={s.mean():.4f}  std={s.std():.4f}")

        # quick separability check via simple threshold
        combined = np.concatenate([b, s])
        labels = np.concatenate([np.ones(len(b)), np.zeros(len(s))])
        best_acc = 0
        for thresh in np.percentile(combined, np.arange(1, 100, 2)):
            pred = (combined > thresh).astype(int)
            acc = max((pred == labels).mean(), (pred != labels).mean())
            best_acc = max(best_acc, acc)
        flag = "  <<< STRONG SHORTCUT" if best_acc > 0.85 else ""
        print(f"  separability using this feature alone: {best_acc*100:.1f}%{flag}\n")


if __name__ == '__main__':
    main()