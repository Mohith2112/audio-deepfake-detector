"""
Streamlit demo for the audio deepfake detector.

Run with:  streamlit run app.py

Lets a user upload an audio clip, see the waveform, get a real/fake
verdict with confidence, and view the Grad-CAM heatmap explaining why.
"""

import os
import sys
import tempfile
import numpy as np
import librosa
import librosa.display
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from model import SpoofCNN
from gradcam import GradCAM, preprocess_audio

SAMPLE_RATE = 16000
MODEL_CHECKPOINT = "best_model.pt"


@st.cache_resource
def load_model():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    model = SpoofCNN().to(device)
    model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
    model.eval()
    return model, device


def predict_with_gradcam(model, device, audio_path):
    target_layer = model.conv_block[12]
    gradcam = GradCAM(model, target_layer)

    mel_db, mel_norm = preprocess_audio(audio_path)
    x = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    cam, prob = gradcam.generate(x)

    cam_resized = np.array(
        F.interpolate(
            torch.tensor(cam).unsqueeze(0).unsqueeze(0),
            size=mel_db.shape, mode='bilinear', align_corners=False
        ).squeeze()
    )

    return mel_db, cam_resized, prob


def main():
    st.set_page_config(page_title="Audio Deepfake Detector", page_icon="🎙️", layout="centered")

    st.title("🎙️ Audio Deepfake Detector")
    st.markdown(
        "Upload an audio clip to check whether it's **genuine human speech** "
        "or **AI-generated / voice-cloned audio**. Trained on the "
        "[ASVspoof 2019 LA](https://www.asvspoof.org/) dataset."
    )

    if not os.path.exists(MODEL_CHECKPOINT):
        st.error(
            f"Model checkpoint `{MODEL_CHECKPOINT}` not found. "
            "Run `train.py` first to produce it."
        )
        return

    model, device = load_model()

    uploaded_file = st.file_uploader(
        "Upload audio", type=["flac", "wav", "mp3", "ogg"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file)

        # waveform display
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE)
        fig_wave, ax_wave = plt.subplots(figsize=(10, 2))
        librosa.display.waveshow(audio, sr=SAMPLE_RATE, ax=ax_wave, color="#4C6EF5")
        ax_wave.set_title("Waveform")
        st.pyplot(fig_wave)
        plt.close(fig_wave)

        with st.spinner("Analyzing..."):
            mel_db, cam, prob = predict_with_gradcam(model, device, tmp_path)

        is_real = prob > 0.5
        confidence = prob if is_real else 1 - prob

        col1, col2 = st.columns(2)
        with col1:
            if is_real:
                st.success("✅ Likely REAL")
            else:
                st.error("🚨 Likely AI-GENERATED")
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")

        st.markdown("### Why the model flagged this")
        st.caption(
            "The heatmap below highlights the time-frequency regions of the "
            "spectrogram that most influenced the verdict."
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        img1 = librosa.display.specshow(
            mel_db, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='mel', ax=axes[0]
        )
        axes[0].set_title("Mel spectrogram")
        fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

        librosa.display.specshow(
            mel_db, sr=SAMPLE_RATE, hop_length=160, x_axis='time', y_axis='mel', ax=axes[1]
        )
        axes[1].imshow(
            cam, aspect='auto', origin='lower', cmap='jet', alpha=0.45,
            extent=axes[1].get_xlim() + axes[1].get_ylim()
        )
        axes[1].set_title("Grad-CAM — model attention")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        os.unlink(tmp_path)

    st.divider()
    st.caption(
        "Model: custom CNN (BatchNorm + global average pooling) trained on "
        "ASVspoof 2019 LA. Metric reported: Equal Error Rate (EER), the "
        "standard benchmark for spoofing detection."
    )


if __name__ == '__main__':
    main()