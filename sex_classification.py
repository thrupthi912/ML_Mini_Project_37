"""
sex_classification.py  — Member 2
-----------------------------------
Classifies each detected fly as 'male' or 'female' based on contour
shape and intensity features.

Member 2 tasks:
    - Load labeled sex data from input/images/sex_labels.csv
    - Choose and train a classifier (e.g. SVM, Random Forest)
    - Evaluate and save model to models/sex_clf_model.joblib
"""

import os
import numpy as np
import joblib


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def sex_features(contour, patch) -> np.ndarray:
    """
    Extract features for sex classification from a contour and its image patch.

    Suggested features:
      - Contour area (females are typically larger)
      - Aspect ratio (body shape differs between sexes)
      - Mean pixel intensity of the patch
      - Standard deviation of pixel intensity

    Parameters
    ----------
    contour : OpenCV contour
    patch   : np.ndarray — cropped grayscale image of the fly

    Returns
    -------
    feat_vec : 1-D float32 array
    """
    import cv2

    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0

    mean_intensity = float(np.mean(patch)) if patch is not None else 128.0
    std_intensity  = float(np.std(patch))  if patch is not None else 0.0

    return np.array([area, aspect_ratio, mean_intensity, std_intensity],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction (used by main.py once model is trained)
# ---------------------------------------------------------------------------

def predict_sex(clf, contour, patch) -> str:
    """
    Predict the sex of a single fly.

    Returns
    -------
    "male" or "female"
    """
    feat_vec = sex_features(contour, patch).reshape(1, -1)
    return clf.predict(feat_vec)[0]


def load_model(path: str = "models/sex_clf_model.joblib"):
    """Load a trained sex classification model."""
    if not os.path.exists(path):
        return None
    return joblib.load(path)


if __name__ == "__main__":
    print("sex_classification.py — Member 2 module.")
    print("Train using labeled data from input/images/sex_labels.csv")
