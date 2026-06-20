"""
Grad-CAM explainability for the spoof detector.

Produces a heatmap over the mel spectrogram showing which time-frequency
regions most influenced the model's "spoof" or "bonafide" decision.

How it works:
- Hook the last conv layer's activations (forward hook) and gradients
  (backward hook).
- Run a forward pass, get the predicted class's logit.
- Backprop from that logit to get gradients w.r.t. the last conv feature
  maps — these gradients tell us "how much would the prediction change if
  this region were more/less active."
- Weight each feature map channel by the average gradient, sum them up,
  ReLU the result -> a coarse heatmap of "what the model looked at."
- Upsample that heatmap to the original spectrogram size and overlay it.

This targets the last conv block in SpoofCNN (128 channels, before
global average pooling) since that's where spatial information is still
preserved — after pooling, the spatial structure is gone.
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from model import SpoofCNN

SAMPLE_RATE = 16000
DURATION = 4.0
N_MELS = 128
MAX_LEN = int(SAMPLE_RATE * DURATION)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        input_tensor: (1, 1, n_mels, T)
        Returns: heatmap (n_mels, T) normalized to [0, 1], and the
                 model's predicted probability of bonafide.
        """
        self.model.zero_grad()
        logit = self.model(input_tensor)          # (1,)
        prob = torch.sigmoid(logit).item()

        # Backprop from the raw logit — this explains "what pushed the
        # score in the direction it ended up", regardless of class.
        logit.backward()

        # gradients/activations: (1, C, H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, prob


def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)

    # Must match dataset.py's trimming exactly — see diagnose_silence.py
    # for why this matters (silence patterns were a classification shortcut)
    audio, _ = librosa.effects.trim(audio, top_db=30)

    if len(audio) > MAX_LEN:
        start = (len(audio) - MAX_LEN) // 2
        audio = audio[start:start + MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)), mode='constant')

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=512, hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db, mel_norm  # raw dB for display, normalized for the model


def explain(audio_path, model_checkpoint='best_model.pt', save_path=None):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    # target the last conv layer (index 12 in conv_block: the final Conv2d
    # before global pooling — see model.py's conv_block Sequential)
    target_layer = model.conv_block[12]
    gradcam = GradCAM(model, target_layer)

    mel_db, mel_norm = preprocess_audio(audio_path)
    x = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x.requires_grad_(False)

    cam, prob = gradcam.generate(x)

    # upsample CAM (small spatial size after 3 poolings) to match spectrogram
    cam_resized = np.array(
        F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=mel_db.shape, mode='bilinear', align_corners=False
        ).squeeze()
    )

    verdict = "BONAFIDE (real)" if prob > 0.5 else "SPOOF (fake)"
    confidence = prob if prob > 0.5 else 1 - prob

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    img1 = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='mel', ax=axes[0]
    )
    axes[0].set_title("Mel spectrogram")
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    img2 = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='mel', ax=axes[1]
    )
    axes[1].imshow(
        cam_resized, aspect='auto', origin='lower', cmap='jet', alpha=0.45,
        extent=axes[1].get_xlim() + axes[1].get_ylim()
    )
    axes[1].set_title(f"Grad-CAM — {verdict} ({confidence*100:.1f}% confidence)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
    return verdict, confidence, cam_resized


if __name__ == '__main__':
    # Example usage — point this at any single flac file to explain it
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', help='Path to a .flac or .wav file to explain')
    parser.add_argument('--checkpoint', default='best_model.pt')
    parser.add_argument('--out', default='gradcam_output.png')
    args = parser.parse_args()

    verdict, confidence, _ = explain(args.audio_path, args.checkpoint, args.out)
    print(f"Verdict: {verdict} | Confidence: {confidence*100:.1f}%")