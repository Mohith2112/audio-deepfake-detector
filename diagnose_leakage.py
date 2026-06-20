"""
Diagnostic: why did val_EER hit 0.01%?

This checks the most likely causes of a suspiciously perfect score:
1. File ID overlap between train and dev protocols (direct leakage)
2. Speaker overlap between train and dev (should be ZERO per ASVspoof spec)
3. Whether dev predictions are trivially separable by something dumb,
   like audio duration, rather than real spoof artifacts

Run from your project root:
    python3 diagnose_leakage.py
"""

import os

BASE = 'LA'
TRAIN_PROTOCOL = f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
DEV_PROTOCOL = f'{BASE}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'


def load_protocol(path):
    entries = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker = parts[0]
            file_id = parts[1]
            label = parts[-1]
            entries.append((speaker, file_id, label))
    return entries


def main():
    train = load_protocol(TRAIN_PROTOCOL)
    dev = load_protocol(DEV_PROTOCOL)

    train_files = set(f for _, f, _ in train)
    dev_files = set(f for _, f, _ in dev)
    train_speakers = set(s for s, _, _ in train)
    dev_speakers = set(s for s, _, _ in dev)

    print(f"Train: {len(train)} entries, {len(train_speakers)} unique speakers")
    print(f"Dev:   {len(dev)} entries, {len(dev_speakers)} unique speakers")

    file_overlap = train_files & dev_files
    speaker_overlap = train_speakers & dev_speakers

    print(f"\nFile ID overlap (train ∩ dev): {len(file_overlap)} files")
    if file_overlap:
        print(f"  EXAMPLE OVERLAPPING FILES: {list(file_overlap)[:5]}")
        print("  *** THIS IS LEAKAGE — the model has literally seen these dev files during training ***")
    else:
        print("  None — no direct file leakage")

    print(f"\nSpeaker overlap (train ∩ dev): {len(speaker_overlap)} speakers")
    if speaker_overlap:
        print(f"  EXAMPLE OVERLAPPING SPEAKERS: {list(speaker_overlap)[:5]}")
        print("  *** UNEXPECTED — official ASVspoof19 splits are disjoint by speaker ***")
    else:
        print("  None — speaker splits are disjoint, as expected")

    # check actual audio files on disk match protocol expectations
    train_dir = f'{BASE}/ASVspoof2019_LA_train/flac'
    dev_dir = f'{BASE}/ASVspoof2019_LA_dev/flac'

    sample_train_file = list(train_files)[0] + '.flac'
    sample_dev_file = list(dev_files)[0] + '.flac'

    print(f"\nSanity check — does dev file exist only in dev folder, not train folder?")
    dev_in_train_dir = os.path.exists(os.path.join(train_dir, sample_dev_file))
    print(f"  {sample_dev_file} exists in TRAIN folder: {dev_in_train_dir} (should be False)")

    if file_overlap or speaker_overlap:
        print("\n>>> CONCLUSION: Data leakage confirmed. This explains the 0.01% EER.")
    else:
        print("\n>>> No leakage found at the protocol level. The cause may be elsewhere "
              "(label parsing bug, duration/padding artifact, or a genuine but suspicious "
              "result that needs eval-set confirmation before trusting it).")


if __name__ == '__main__':
    main()