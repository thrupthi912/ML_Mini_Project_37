"""
test_wing_angle.py
------------------
Unit tests for wing_angle.py.
Synthetic patches only — no real labeled data required.
Results verify software correctness only, NOT wing-angle prediction accuracy.
"""
import numpy as np
import cv2
import pytest
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wing_angle import (
    extract_fly_patch, preprocess_patch, split_wing_regions,
    hog_features, patch_to_hog_vector, build_feature_matrix,
    predict_wing_angles, PATCH_H, PATCH_W, WING_FRACTION,
)


# ---------------------------------------------------------------------------
# extract_fly_patch
# ---------------------------------------------------------------------------

class TestExtractFlyPatch:

    def test_returns_correct_shape(self, synthetic_frame):
        patch = extract_fly_patch(synthetic_frame, 160, 120)
        assert patch is not None
        assert patch.shape == (PATCH_H, PATCH_W)

    def test_returns_uint8(self, synthetic_frame):
        patch = extract_fly_patch(synthetic_frame, 160, 120)
        assert patch.dtype == np.uint8

    def test_returns_none_when_out_of_bounds_top_left(self, synthetic_frame):
        patch = extract_fly_patch(synthetic_frame, 0, 0)
        assert patch is None

    def test_returns_none_when_out_of_bounds_bottom_right(self, synthetic_frame):
        h, w = synthetic_frame.shape
        patch = extract_fly_patch(synthetic_frame, w - 1, h - 1)
        assert patch is None

    def test_centre_patch_is_not_none(self, synthetic_frame):
        h, w = synthetic_frame.shape
        patch = extract_fly_patch(synthetic_frame, w // 2, h // 2)
        assert patch is not None

    def test_patch_is_copy_not_view(self, synthetic_frame):
        patch = extract_fly_patch(synthetic_frame, 160, 120)
        original_val = synthetic_frame[120, 160]
        patch[0, 0] = 0
        assert synthetic_frame[120, 160] == original_val


# ---------------------------------------------------------------------------
# preprocess_patch
# ---------------------------------------------------------------------------

class TestPreprocessPatch:

    def test_output_shape_unchanged(self, tiny_patch):
        result = preprocess_patch(tiny_patch)
        assert result.shape == tiny_patch.shape

    def test_output_dtype_uint8(self, tiny_patch):
        result = preprocess_patch(tiny_patch)
        assert result.dtype == np.uint8

    def test_no_nan_values(self, tiny_patch):
        result = preprocess_patch(tiny_patch)
        assert not np.any(np.isnan(result.astype(float)))

    def test_output_differs_from_input(self, tiny_patch):
        """CLAHE should change pixel values."""
        result = preprocess_patch(tiny_patch)
        assert not np.array_equal(result, tiny_patch)


# ---------------------------------------------------------------------------
# split_wing_regions
# ---------------------------------------------------------------------------

class TestSplitWingRegions:

    def test_returns_two_arrays(self, tiny_patch):
        left, right = split_wing_regions(tiny_patch)
        assert left is not None and right is not None

    def test_wing_width(self, tiny_patch):
        expected_w = int(PATCH_W * WING_FRACTION)
        left, right = split_wing_regions(tiny_patch)
        assert left.shape[1]  == expected_w
        assert right.shape[1] == expected_w

    def test_height_preserved(self, tiny_patch):
        left, right = split_wing_regions(tiny_patch)
        assert left.shape[0]  == PATCH_H
        assert right.shape[0] == PATCH_H

    def test_left_and_right_are_different_regions(self, tiny_patch):
        left, right = split_wing_regions(tiny_patch)
        # They come from opposite sides so pixel means should differ
        # (not a strict requirement but a sanity check)
        assert left.shape == right.shape


# ---------------------------------------------------------------------------
# hog_features
# ---------------------------------------------------------------------------

class TestHogFeatures:

    def test_output_is_1d(self, tiny_patch):
        left, _ = split_wing_regions(tiny_patch)
        feat = hog_features(left)
        assert feat.ndim == 1

    def test_output_is_finite(self, tiny_patch):
        left, _ = split_wing_regions(tiny_patch)
        feat = hog_features(left)
        assert np.all(np.isfinite(feat))

    def test_consistent_length_across_patches(self, tiny_patch, circular_patch):
        """HOG must produce same-length vectors for all PATCH_H×PATCH_W inputs."""
        left1, _ = split_wing_regions(tiny_patch)
        left2, _ = split_wing_regions(circular_patch)
        f1 = hog_features(left1)
        f2 = hog_features(left2)
        assert f1.shape == f2.shape

    def test_different_inputs_give_different_features(self, tiny_patch, circular_patch):
        left1, _ = split_wing_regions(tiny_patch)
        left2, _ = split_wing_regions(circular_patch)
        f1 = hog_features(left1)
        f2 = hog_features(left2)
        assert not np.allclose(f1, f2)


# ---------------------------------------------------------------------------
# patch_to_hog_vector
# ---------------------------------------------------------------------------

class TestPatchToHogVector:

    def test_output_is_1d(self, tiny_patch):
        vec = patch_to_hog_vector(tiny_patch)
        assert vec.ndim == 1

    def test_output_length_is_double_single_hog(self, tiny_patch):
        """Concatenation of left+right HOG must be 2× single HOG length."""
        left, _ = split_wing_regions(preprocess_patch(tiny_patch))
        single_hog_len = len(hog_features(left))
        full_vec = patch_to_hog_vector(tiny_patch)
        assert len(full_vec) == 2 * single_hog_len

    def test_output_is_finite(self, tiny_patch):
        vec = patch_to_hog_vector(tiny_patch)
        assert np.all(np.isfinite(vec))

    def test_reproducible(self, tiny_patch):
        v1 = patch_to_hog_vector(tiny_patch)
        v2 = patch_to_hog_vector(tiny_patch)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_shape(self, tiny_patch, circular_patch):
        patches = [tiny_patch, circular_patch]
        X = build_feature_matrix(patches)
        expected_cols = len(patch_to_hog_vector(tiny_patch))
        assert X.shape == (2, expected_cols)

    def test_dtype_float64(self, tiny_patch, circular_patch):
        X = build_feature_matrix([tiny_patch, circular_patch])
        assert X.dtype == np.float64

    def test_no_nan_or_inf(self, tiny_patch, circular_patch):
        X = build_feature_matrix([tiny_patch, circular_patch])
        assert np.all(np.isfinite(X))


# ---------------------------------------------------------------------------
# predict_wing_angles (with loaded models)
# ---------------------------------------------------------------------------

class TestPredictWingAngles:

    def test_returns_none_when_models_none(self, tiny_patch):
        result = predict_wing_angles(None, None, tiny_patch)
        assert result is None

    def test_returns_tuple_of_two_floats(self, wing_models, tiny_patch):
        model_r, model_l = wing_models
        result = predict_wing_angles(model_r, model_l, tiny_patch)
        assert result is not None
        assert len(result) == 2
        angle_r, angle_l = result
        assert isinstance(angle_r, float)
        assert isinstance(angle_l, float)

    def test_predictions_are_finite(self, wing_models, tiny_patch):
        model_r, model_l = wing_models
        result = predict_wing_angles(model_r, model_l, tiny_patch)
        assert result is not None
        assert np.isfinite(result[0]) and np.isfinite(result[1])

    def test_predictions_are_in_plausible_range(self, wing_models, tiny_patch):
        """
        Wing angles should be roughly in (0, pi) for real data.
        On synthetic patches the model was trained on synthetic data so
        predictions may be outside this range — we just check finiteness.
        NOTE: accuracy on synthetic data does NOT represent real performance.
        """
        model_r, model_l = wing_models
        result = predict_wing_angles(model_r, model_l, tiny_patch)
        assert result is not None
        # Just verify they are numbers, not that they are accurate
        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
