---
title: Audio Deepfake Detector
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
---

# 🎙️ Audio Deepfake Detector

A CNN-based system that distinguishes genuine human speech from AI-generated / voice-cloned audio, trained and evaluated on the [ASVspoof 2019 LA](https://www.asvspoof.org/) benchmark dataset.

**[Live demo →](https://huggingface.co/spaces/Mickeylabs/audio-deepfake-detector)**

---

## Why this project

Voice cloning tools have made synthetic speech nearly indistinguishable from real recordings — a real threat to phone-based authentication, media integrity, and fraud prevention. This project builds and evaluates a spoofing countermeasure: given an audio clip, predict whether it's genuine ("bonafide") or machine-generated ("spoof"), and explain *why* using Grad-CAM.

## Results

| Metric | Value | Notes |
|---|---|---|
| **Eval-set EER (reportable)** | **23.93%** | Measured on the full official eval set (71,237 files), including spoofing attack types never seen during training |
| Dev-set EER (during training) | 0.47% | Measured on attack types seen during training/tuning |
| Dataset | ASVspoof 2019 LA | 25,380 train / 24,844 dev / 71,237 eval, official disjoint speaker splits |

### Why there's a gap between dev and eval EER

ASVspoof 2019 LA is deliberately designed so the eval set includes spoofing attack types that **never appear in train or dev** — this "known vs unknown attack" generalization gap is the core challenge the benchmark is built around, not a flaw unique to this model. The official baseline systems show the same pattern: strong performance on known attacks, a meaningful drop on unseen synthesis methods. This model learned the training attacks' artifacts very well (0.47% EER) but generalizes less completely to attack types it never encountered, landing at 23.93% EER on the full eval set.

**Future work to close this gap:** fine-tuning a pretrained self-supervised speech model (e.g. Wav2Vec2) instead of training a CNN from scratch tends to generalize better to unseen synthesis artifacts, since it starts from representations learned on a much broader speech distribution.

> ⚠️ An earlier version of this model reported a suspiciously perfect ~0.01% EER. Investigation traced this to a data leak: bonafide and spoof files in this dataset have systematically different leading-silence durations (~1.15s vs ~0.65s) and overall silence percentage (~47% vs ~30%), letting the model classify using a non-content artifact instead of real spoofing cues. Fixed by trimming silence consistently (`librosa.effects.trim`) before computing spectrograms, in both training and inference. See `diagnose_silence.py` and `diagnose_predictions.py` for the diagnostic process.

## How it works

```
Raw audio (.flac)
      │
      ▼
Mel spectrogram (128 mels, log-scale, normalized)
      │
      ▼
CNN  (4 conv blocks, BatchNorm, global average pooling)
      │
      ▼
Sigmoid → P(bonafide)
      │
      ▼
Grad-CAM → heatmap of which time-frequency regions drove the decision
```

### Why this architecture, not a bigger one

An earlier version of this project used a `Flatten → Dense(48000→128)` head with no batch normalization. It hit 99% training accuracy while validation accuracy collapsed to ~20% — classic overfitting driven by an oversized parameter count relative to the ~25k training files, with no normalization to stabilize learning. This version fixes that with:

- **BatchNorm after every conv layer** — keeps activations stable as the network trains
- **Global average pooling instead of Flatten** — removes ~90% of the parameter count in the classifier head
- **Class-weighted sampling** — corrects for the dataset's natural ~9:1 spoof:bonafide imbalance, which otherwise lets a model "cheat" by mostly predicting spoof
- **Early stopping on EER, not accuracy** — accuracy is misleading on an imbalanced dataset; EER is the metric every ASVspoof paper actually reports

## Explainability

Every prediction comes with a Grad-CAM heatmap overlaid on the mel spectrogram, highlighting which time-frequency regions most influenced the verdict — instead of a black-box confidence score.

**Correct prediction** (`LA_E_5849185`, true label: bonafide → predicted: bonafide, 98.9% confidence)

![Grad-CAM correct example](gradcam_bonafide.png)

**A failure case** (`LA_E_2834763`, true label: spoof (attack A11) → predicted: bonafide, 97.5% confidence)

![Grad-CAM failure example](gradcam_spoof.png)

Showing a misclassification alongside a correct one is intentional — at 23.93% eval EER, roughly 1 in 4 borderline cases gets misclassified, and this example is one of them. The heatmap on the failure case shows the model is still attending to plausible regions of the spectrogram; it simply didn't pick up on this particular attack type's (A11) synthesis artifacts, consistent with the known-vs-unknown attack generalization gap discussed above.

## Project structure

```
audio-deepfake-detector/
├── LA/                  # ASVspoof2019 LA dataset (not included — see Setup)
├── dataset.py            # PyTorch Dataset with correct protocol parsing
├── model.py               # CNN architecture
├── metrics.py             # EER computation
├── train.py                # Training loop with weighted sampling + early stopping
├── evaluate.py             # Full evaluation on the eval set + plots
├── gradcam.py              # Grad-CAM explainability
├── app.py                  # Streamlit demo
└── best_model.pt            # Trained model checkpoint
```

## Setup

```bash
git clone <your-repo-url>
cd audio-deepfake-detector

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install torch torchaudio librosa pandas numpy tqdm scikit-learn matplotlib seaborn streamlit
```

### Dataset

Download the ASVspoof 2019 LA partition from the [official source](https://datashare.ed.ac.uk/handle/10283/3336) and unzip it so the `LA/` folder sits in the project root.

### Train

```bash
python3 train.py
```

Produces `best_model.pt`, checkpointed automatically whenever validation EER improves.

### Evaluate

```bash
python3 evaluate.py
```

Runs inference on the full eval set and saves confusion matrix, ROC, and PR curve plots.

### Explain a single clip

```bash
python3 gradcam.py path/to/audio.flac --out result.png
```

### Run the demo

```bash
streamlit run app.py
```

## Dataset citation

```bibtex
@InProceedings{Todisco2019,
  title     = {ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection},
  author    = {Todisco, Massimiliano and Wang, Xin and Sahidullah, Md and Delgado, H{\'e}ctor
               and Nautsch, Andreas and Yamagishi, Junichi and Evans, Nicholas and Kinnunen, Tomi
               and Lee, Kong Aik},
  booktitle = {Proc. Interspeech 2019},
  year      = {2019}
}
```

## License

This project's code is provided as-is for portfolio/educational use. The ASVspoof 2019 dataset is licensed separately under [ODC-BY](https://opendatacommons.org/licenses/by/1.0/index.html).