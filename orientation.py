"""
orientation.py  — Member 2
----------------------------
Estimates the body orientation angle (in radians) of each detected fly
using image moments on the fly's binary mask.

Member 2 tasks:
    - Load labeled orientation data from input/images/orientation_labels.csv
    - Improve or replace estimate_orientation() with an ML-based approach
    - Evaluate and save model to models/orientation_model.joblib
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Geometry-based orientation (baseline — no training required)
# ---------------------------------------------------------------------------

def estimate_orientation(patch: np.ndarray) -> float:
    """
    Estimate the fly's body orientation using image moments.

    The principal axis of the binary blob gives the body orientation angle.

    Parameters
    ----------
    patch : grayscale uint8 image of the fly region

    Returns
    -------
    angle : float — orientation in radians [0, pi)
    """
    _, binary = cv2.threshold(patch, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    M = cv2.moments(binary)
    if abs(M["mu20"] - M["mu02"]) < 1e-6 and abs(M["mu11"]) < 1e-6:
        return 0.0

    angle = 0.5 * np.arctan2(2 * M["mu11"],
                              M["mu20"] - M["mu02"])
    return float(angle)


# ---------------------------------------------------------------------------
# Prediction (used by main.py)
# ---------------------------------------------------------------------------

def predict_orientation(patch: np.ndarray, model=None) -> float:
    """
    Predict orientation angle for a fly patch.
    Uses the geometry baseline if no trained model is provided.

    Returns
    -------
    angle : float (radians)
    """
    if model is not None:
        # Member 2: replace with model.predict(features)
        pass
    return estimate_orientation(patch)


if __name__ == "__main__":
    print("orientation.py — Member 2 module.")
    print("Baseline (moments-based) estimate_orientation() is functional.")
    print("Replace with ML model using input/images/orientation_labels.csv")
