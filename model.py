"""
CNN architecture for spoof detection on mel spectrograms.

Key fixes vs the original notebook:
- BatchNorm after every conv (the original had none — this is a big reason
  training accuracy shot up while val collapsed: no normalization to keep
  activations stable as the network got confident)
- Global average pooling instead of Flatten + huge Dense layer
  (Flatten on a (32, 16, 27) feature map -> a 13000+ unit Dense layer is a
  massive parameter count relative to ~25k training files — that alone
  is enough to memorize the train set)
- Single sigmoid output (binary task) instead of 2-unit softmax
- Dropout tuned down slightly + spatial dropout on conv layers
"""

import torch
import torch.nn as nn


class SpoofCNN(nn.Module):
    def __init__(self, n_mels=128, dropout=0.35):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # single logit, use BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x.squeeze(1)  # (batch,)