"""
Diagnostic: is the model learning a duration/padding shortcut instead of
real spoof artifacts?

Checks:
1. Raw audio duration distribution for bonafide vs spoof in the dev set
   -- if these are cleanly separable just by length, that's a huge red flag
2. How much of each class ends up needing padding vs cropping, given
   our fixed 4-second window
3. A trivial "classifier": predict bonafide/spoof using ONLY duration,
   see what accuracy that alone gets. If this matches your model's
   suspiciously high accuracy, duration is very likely the leak.

Run from your project root:
    python3 diagnose_duration.py
"""

import os
import librosa
import numpy as np

BASE = 'LA'
DEV_PROTOCOL = f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
DEV_AUDIO_DIR = f'{BASE}/ASVspoof2019_LA_dev/flac'

SAMPLE_RATE = 16000
N_SAMPLE = 500  # don't need all 24k files to see the pattern


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
    entries = load_protocol(DEV_PROTOCOL)

    bonafide = [e for e in entries if e[1] == 1]
    spoof = [e for e in entries if e[1] == 0]

    # sample roughly proportionally, but cap total work
    n_bona = min(len(bonafide), N_SAMPLE // 2)
    n_spoof = min(len(spoof), N_SAMPLE // 2)

    np.random.seed(42)
    sample = (
        list(np.random.choice(len(bonafide), n_bona, replace=False))
    )
    bona_sample = [bonafide[i] for i in sample]
    sample2 = list(np.random.choice(len(spoof), n_spoof, replace=False))
    spoof_sample = [spoof[i] for i in sample2]

    print(f"Sampling {len(bona_sample)} bonafide + {len(spoof_sample)} spoof files for duration check...\n")

    def get_durations(file_list):
        durations = []
        for file_id, _ in file_list:
            path = os.path.join(DEV_AUDIO_DIR, file_id + '.flac')
            if not os.path.exists(path):
                continue
            dur = librosa.get_duration(path=path)
            durations.append(dur)
        return np.array(durations)

    bona_durs = get_durations(bona_sample)
    spoof_durs = get_durations(spoof_sample)

    print("=== Duration statistics (seconds) ===")
    print(f"Bonafide: mean={bona_durs.mean():.3f}  std={bona_durs.std():.3f}  "
          f"min={bona_durs.min():.3f}  max={bona_durs.max():.3f}")
    print(f"Spoof:    mean={spoof_durs.mean():.3f}  std={spoof_durs.std():.3f}  "
          f"min={spoof_durs.min():.3f}  max={spoof_durs.max():.3f}")

    # how separable are they by duration ALONE using a simple midpoint threshold?
    combined = np.concatenate([bona_durs, spoof_durs])
    labels = np.concatenate([np.ones(len(bona_durs)), np.zeros(len(spoof_durs))])

    best_acc = 0
    best_thresh = None
    for thresh in np.percentile(combined, np.arange(1, 100, 1)):
        pred = (combined > thresh).astype(int)
        acc = max((pred == labels).mean(), (pred != labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"\n=== Trivial duration-only classifier ===")
    print(f"Best single-threshold accuracy using ONLY duration: {best_acc*100:.2f}%")
    print(f"(threshold: {best_thresh:.3f}s)")

    if best_acc > 0.90:
        print("\n*** RED FLAG: duration alone almost perfectly separates the classes. ***")
        print("*** Your model's near-0% EER is very likely driven by duration/padding ***")
        print("*** patterns rather than learned spoof artifacts. ***")
    elif best_acc > 0.75:
        print("\n*** Duration is a meaningfully informative shortcut, partially explains the result. ***")
    else:
        print("\nDuration alone is not a strong separator — the leak is likely elsewhere "
              "(check padding/silence patterns, or label parsing).")


if __name__ == '__main__':
    main()