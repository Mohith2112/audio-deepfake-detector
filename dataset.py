"""
Dataset loading for ASVspoof 2019 LA.

Key fixes vs the original notebook:
- Correct protocol column parsing (speaker, file_id, -, system_id, label)
- Uses the OFFICIAL train/dev/eval splits instead of slicing one folder 80/20
  (slicing one folder leaks near-duplicate speech into both train and val)
- Exposes per-sample labels so a WeightedRandomSampler can fix class imbalance
- Pads/truncates by waveform length (not spectrogram width) for consistency
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    def __init__(self, protocol_path, audio_dir, sample_rate=16000,
                 duration=4.0, n_mels=128, augment=False):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * duration)
        self.n_mels = n_mels
        self.augment = augment

        self.file_ids = []
        self.labels = []  # 0 = spoof, 1 = bonafide

        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Official ASVspoof2019 LA cm protocol format:
                # speaker_id  file_id  -  system_id  label
                file_id = parts[1]
                label_str = parts[-1]
                label = 1 if label_str == 'bonafide' else 0
                self.file_ids.append(file_id)
                self.labels.append(label)

        if len(self.file_ids) == 0:
            raise ValueError(
                f"No entries parsed from {protocol_path}. "
                f"Check the protocol file format matches expected columns."
            )

    def __len__(self):
        return len(self.file_ids)

    def _load_and_fix_length(self, path):
        audio, _ = librosa.load(path, sr=self.sample_rate)

        # CRITICAL FIX: trim leading/trailing silence before anything else.
        # Diagnosis found bonafide files average ~1.15s leading silence vs
        # ~0.65s for spoof, and ~47% vs ~30% total silence — a CNN can (and
        # did) learn to classify using this non-content artifact alone,
        # producing a near-0% EER that does not reflect real generalization.
        # Trimming forces the model to learn from actual voiced content.
        audio, _ = librosa.effects.trim(audio, top_db=30)

        if self.augment:
            # light augmentation only on training data
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.003, audio.shape)
                audio = audio + noise
            if np.random.rand() < 0.3:
                shift = np.random.randint(-1600, 1600)
                audio = np.roll(audio, shift)

        if len(audio) > self.max_len:
            # random crop during training, center crop during eval
            if self.augment:
                start = np.random.randint(0, len(audio) - self.max_len)
            else:
                start = (len(audio) - self.max_len) // 2
            audio = audio[start:start + self.max_len]
        else:
            pad = self.max_len - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')

        return audio

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]
        path = os.path.join(self.audio_dir, file_id + '.flac')

        audio = self._load_and_fix_length(path)

        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels, n_fft=512, hop_length=160
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # normalize to roughly [-1, 1] — stabilizes CNN training
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        return mel_tensor, torch.tensor(label, dtype=torch.long)

    def get_class_counts(self):
        spoof = sum(1 for l in self.labels if l == 0)
        bonafide = sum(1 for l in self.labels if l == 1)
        return {'spoof': spoof, 'bonafide': bonafide}