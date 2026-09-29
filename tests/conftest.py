"""
conftest.py — shared pytest fixtures for all test modules.

All fixtures use only synthetic data so tests run without the real dataset.
Results from these tests verify software behaviour only, NOT scientific performance.
"""
import sys
import os
import numpy as np
import cv2
import pytest

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Synthetic frame / video fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_frame():
    """
    320×240 grayscale frame with two dark ellipses on a light background.
    Matches the synthetic video generator in preprocessing.py.
    """
    frame = np.full((240, 320), 200, dtype=np.uint8)
    cv2.ellipse(frame, (80,  120), (12, 7), 0, 0, 360, 30, -1)   # fly 1
    cv2.ellipse(frame, (240, 100), (12, 7), 0, 0, 360, 30, -1)   # fly 2
    return frame


@pytest.fixture
def synthetic_background():
    """Plain background — same as frame but without flies."""
    return np.full((240, 320), 200, dtype=np.uint8)


@pytest.fixture
def single_blob_frame():
    """Frame with exactly one fly-sized blob."""
    frame = np.full((240, 320), 200, dtype=np.uint8)
    cv2.ellipse(frame, (160, 120), (12, 7), 0, 0, 360, 30, -1)
    return frame


@pytest.fixture
def empty_frame():
    """Completely uniform frame — no flies."""
    return np.full((240, 320), 200, dtype=np.uint8)


@pytest.fixture
def tiny_patch():
    """64×64 patch with a dark ellipse (fly body) for orientation/wing tests."""
    patch = np.full((64, 64), 200, dtype=np.uint8)
    cv2.ellipse(patch, (32, 32), (22, 7), 35, 0, 360, 30, -1)
    return patch


@pytest.fixture
def circular_patch():
    """64×64 patch with a circular blob — degenerate orientation case."""
    patch = np.full((64, 64), 200, dtype=np.uint8)
    cv2.circle(patch, (32, 32), 15, 30, -1)
    return patch


@pytest.fixture
def synthetic_contour_one():
    """A single contour matching a one-fly blob (~265 px²)."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(mask, (160, 120), (12, 7), 0, 0, 360, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


@pytest.fixture
def synthetic_contour_noise():
    """A tiny contour that should be classified as 'zero' (noise)."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 3, 255, -1)   # ~28 px² — below noise threshold
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


@pytest.fixture
def fly_count_model():
    """Load the saved FlyCount model if it exists; skip test if not."""
    model_path = os.path.join(ROOT, "models", "fly_count_model.joblib")
    if not os.path.isfile(model_path):
        pytest.skip("fly_count_model.joblib not found — run: python fly_count.py")
    from fly_count import load_model
    clf, le = load_model(model_path)
    return clf, le


@pytest.fixture
def wing_models():
    """Load wing angle models if they exist; skip if not."""
    r_path = os.path.join(ROOT, "models", "wing_angle_right_model.joblib")
    l_path = os.path.join(ROOT, "models", "wing_angle_left_model.joblib")
    if not os.path.isfile(r_path) or not os.path.isfile(l_path):
        pytest.skip("Wing angle models not found — run: python wing_angle.py")
    from wing_angle import load_wing_models
    return load_wing_models(os.path.join(ROOT, "models"))
