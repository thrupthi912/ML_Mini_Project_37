"""
test_orientation.py
-------------------
Unit tests for orientation.py.
Synthetic patches only — no real labeled data required.
"""
import numpy as np
import cv2
import pytest
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from orientation import moments_orientation, extract_hog_features, predict_orientation


class TestMomentsOrientation:

    def test_horizontal_ellipse_near_zero(self):
        patch = np.full((64, 64), 200, dtype=np.uint8)
        cv2.ellipse(patch, (32, 32), (25, 5), 0, 0, 360, 30, -1)
        angle = moments_orientation(patch)
        assert abs(np.degrees(angle)) < 5, f"Horizontal ellipse gave {np.degrees(angle):.1f}°"

    def test_vertical_ellipse_near_90(self):
        patch = np.full((64, 64), 200, dtype=np.uint8)
        cv2.ellipse(patch, (32, 32), (5, 25), 0, 0, 360, 30, -1)
        angle = moments_orientation(patch)
        # moments returns angle in (-pi/2, pi/2); vertical = ±90°
        assert abs(abs(np.degrees(angle)) - 90) < 5, f"Vertical ellipse gave {np.degrees(angle):.1f}°"

    def test_45_degree_ellipse(self):
        patch = np.full((64, 64), 200, dtype=np.uint8)
        cv2.ellipse(patch, (32, 32), (25, 5), 45, 0, 360, 30, -1)
        angle = moments_orientation(patch)
        assert abs(abs(np.degrees(angle)) - 45) < 5, f"45° ellipse gave {np.degrees(angle):.1f}°"

    def test_circular_blob_returns_zero(self, circular_patch):
        angle = moments_orientation(circular_patch)
        assert abs(angle) < 0.01, f"Circular blob gave {np.degrees(angle):.2f}°, expected 0.0°"

    def test_empty_patch_returns_zero(self, empty_frame):
        patch = empty_frame[:64, :64]
        angle = moments_orientation(patch)
        assert abs(angle) < 0.01

    def test_returns_float(self, tiny_patch):
        angle = moments_orientation(tiny_patch)
        assert isinstance(angle, float)

    def test_result_in_valid_range(self, tiny_patch):
        """Moments angle must be in (-pi/2, pi/2)."""
        angle = moments_orientation(tiny_patch)
        assert -np.pi / 2 <= angle <= np.pi / 2, f"Angle {angle:.3f} out of range"

    def test_180_degree_ambiguity_documented(self, tiny_patch):
        """
        Verify that both theta and theta+pi give the same moments result.
        This is the documented 180° ambiguity of image moments.
        """
        angle = moments_orientation(tiny_patch)
        # Rotating the patch 180° should give the same axis angle
        rotated = cv2.rotate(tiny_patch, cv2.ROTATE_180)
        angle_rotated = moments_orientation(rotated)
        # The axis angle should be very close (within 5°)
        diff = abs(np.degrees(angle) - np.degrees(angle_rotated))
        diff = min(diff, 180 - diff)  # account for ±180° equivalence
        assert diff < 5, f"180° rotated patch gave different axis: {diff:.1f}° difference"


class TestExtractHogFeatures:

    def test_output_is_1d(self, tiny_patch):
        feat = extract_hog_features(tiny_patch)
        assert feat.ndim == 1

    def test_output_dtype_float(self, tiny_patch):
        feat = extract_hog_features(tiny_patch)
        assert feat.dtype in (np.float32, np.float64)

    def test_consistent_length(self, tiny_patch, circular_patch):
        """HOG features must have the same length for all patches."""
        f1 = extract_hog_features(tiny_patch)
        f2 = extract_hog_features(circular_patch)
        assert f1.shape == f2.shape, "HOG feature length inconsistent across patches"

    def test_no_nan_or_inf(self, tiny_patch):
        feat = extract_hog_features(tiny_patch)
        assert np.all(np.isfinite(feat))

    def test_different_patches_give_different_features(self, tiny_patch, circular_patch):
        f1 = extract_hog_features(tiny_patch)
        f2 = extract_hog_features(circular_patch)
        assert not np.allclose(f1, f2), "Different patches must give different HOG features"


class TestPredictOrientation:

    def test_without_model_returns_float(self, tiny_patch):
        angle = predict_orientation(tiny_patch, model=None)
        assert isinstance(angle, float)

    def test_without_model_returns_same_as_moments(self, tiny_patch):
        angle_pred = predict_orientation(tiny_patch, model=None)
        angle_mom  = moments_orientation(tiny_patch)
        assert abs(angle_pred - angle_mom) < 1e-9

    def test_result_is_finite(self, tiny_patch):
        angle = predict_orientation(tiny_patch, model=None)
        assert np.isfinite(angle)
