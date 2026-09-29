"""
orientation.py  — Person 3
---------------------------
Estimates the body orientation angle (in radians) of each detected fly
using image moments or PCA on the fly's binary mask.

This is a placeholder skeleton for Person 3 to implement.
The interface is kept consistent with the rest of the pipeline.

TODO (Person 3):
    - Load labeled orientation data from input/images/orientation_labels.csv
    - Implement or improve estimate_orientation() using moments / ML
    - Evaluate and save model to models/orientation_model.joblib
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Geometry-based orientation (no training required — baseline)
# ---------------------------------------------------------------------------

def estimate_orientation(patch: np.ndarray) -> float:
    """
    Estimate the fly's body orientation using image moments (PCA approach).

    The principal axis of the binary blob gives the body orientation angle.

    Parameters
    ----------
    patch : grayscale uint8 image of the fly region

    Returns
    -------
    angle : float — orientation in radians [0, pi)
    """
    # Threshold the patch to get a binary blob
    _, binary = cv2.threshold(patch, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Image moments
    M = cv2.moments(binary)
    if abs(M["mu20"] - M["mu02"]) < 1e-6 and abs(M["mu11"]) < 1e-6:
        return 0.0   # Can't determine orientation — return 0

    # Orientation from central moments
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
        # Person 3: replace with model.predict(features)
        pass
    return estimate_orientation(patch)


# ---------------------------------------------------------------------------
# Placeholder main (Person 3 to implement fully)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("orientation.py — skeleton placeholder for Person 3.")
    print("Implement ML-based orientation estimator here.")
    print("\nBaseline (moments-based) estimate_orientation() is already functional.")
