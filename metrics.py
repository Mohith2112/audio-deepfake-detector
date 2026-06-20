"""
Equal Error Rate (EER) computation.

This is the standard metric for spoofing detection, used in every
ASVspoof paper. Accuracy alone is misleading on this dataset because
of the ~9:1 spoof:bonafide imbalance — a model that predicts "spoof"
for everything still scores ~90% accuracy.

EER = the point where the false acceptance rate (accepting a spoof as
real) equals the false rejection rate (rejecting real audio as spoof).
Lower is better. State-of-the-art systems on ASVspoof19 LA get ~1-5%.
A reasonable CNN baseline typically lands around 8-15%.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_scores):
    """
    y_true: array of 0/1 labels (1 = bonafide)
    y_scores: array of model confidence scores for the bonafide class
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # EER is where fpr == fnr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    return eer, eer_threshold