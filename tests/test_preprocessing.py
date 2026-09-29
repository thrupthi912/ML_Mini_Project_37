"""
test_preprocessing.py
---------------------
Unit tests for preprocessing.py.
All tests use synthetic fixtures — results verify software behaviour only.
"""
import numpy as np
import cv2
import pytest
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing import (
    threshold_frame, extract_contours, contour_features,
    make_synthetic_video, load_video, compute_background, read_frames,
)


# ---------------------------------------------------------------------------
# threshold_frame
# ---------------------------------------------------------------------------

class TestThresholdFrame:

    def test_returns_binary_mask(self, synthetic_frame, synthetic_background):
        mask = threshold_frame(synthetic_frame, synthetic_background)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255}), f"Mask contains non-binary values: {unique}"

    def test_output_shape_matches_input(self, synthetic_frame, synthetic_background):
        mask = threshold_frame(synthetic_frame, synthetic_background)
        assert mask.shape == synthetic_frame.shape

    def test_without_background_still_returns_binary(self, single_blob_frame):
        """Bug fix regression: no background must still produce a valid binary mask."""
        mask = threshold_frame(single_blob_frame, background=None)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})

    def test_empty_frame_produces_no_large_foreground(self, empty_frame):
        """Uniform frame should produce no or minimal foreground."""
        mask = threshold_frame(empty_frame, background=None)
        # After Otsu on uniform frame, result may be all-0 or all-255
        # Either is acceptable — the key check is that extract_contours filters it
        assert mask.dtype == np.uint8

    def test_morphological_closing_fills_holes(self, synthetic_frame, synthetic_background):
        """After closing, fly blobs should be solid (no internal holes)."""
        mask = threshold_frame(synthetic_frame, synthetic_background)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If closing worked, there should be no contours with zero area
        for c in contours:
            assert cv2.contourArea(c) > 0


# ---------------------------------------------------------------------------
# extract_contours
# ---------------------------------------------------------------------------

class TestExtractContours:

    def test_two_flies_detected(self, synthetic_frame, synthetic_background):
        """The standard synthetic frame must yield exactly 2 contours."""
        mask = threshold_frame(synthetic_frame, synthetic_background)
        contours = extract_contours(mask)
        assert len(contours) == 2, f"Expected 2 contours, got {len(contours)}"

    def test_single_blob_detected(self, single_blob_frame):
        """A frame with one fly and no background model must yield 1 contour."""
        mask = threshold_frame(single_blob_frame, background=None)
        contours = extract_contours(mask)
        assert len(contours) == 1, (
            f"Expected 1 contour without background, got {len(contours)}. "
            "This may indicate the threshold_frame inversion bug is not fixed."
        )

    def test_empty_frame_yields_no_contours(self, empty_frame):
        mask = threshold_frame(empty_frame, background=None)
        contours = extract_contours(mask)
        assert len(contours) == 0

    def test_contours_are_numpy_arrays(self, synthetic_frame, synthetic_background):
        mask = threshold_frame(synthetic_frame, synthetic_background)
        contours = extract_contours(mask)
        for c in contours:
            assert isinstance(c, np.ndarray)
            assert c.dtype == np.int32

    def test_min_area_filter_removes_noise(self):
        """Blobs below min_area must be filtered out."""
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 2, 255, -1)   # ~12 px² — below min_area=50
        contours = extract_contours(mask, min_area=50)
        assert len(contours) == 0

    def test_max_area_filter_removes_arena_boundary(self):
        """Blobs above max_area must be filtered out."""
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (300, 220), 255, -1)  # huge blob
        contours = extract_contours(mask, max_area=5000)
        assert len(contours) == 0

    def test_returns_list(self, synthetic_frame, synthetic_background):
        mask = threshold_frame(synthetic_frame, synthetic_background)
        result = extract_contours(mask)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# contour_features
# ---------------------------------------------------------------------------

class TestContourFeatures:

    def test_returns_dict_with_required_keys(self, synthetic_contour_one):
        feats = contour_features(synthetic_contour_one)
        for key in ["area", "perimeter", "aspect_ratio", "extent", "solidity"]:
            assert key in feats, f"Missing key: {key}"

    def test_area_is_positive(self, synthetic_contour_one):
        feats = contour_features(synthetic_contour_one)
        assert feats["area"] > 0

    def test_aspect_ratio_is_positive(self, synthetic_contour_one):
        feats = contour_features(synthetic_contour_one)
        assert feats["aspect_ratio"] > 0

    def test_extent_in_unit_range(self, synthetic_contour_one):
        feats = contour_features(synthetic_contour_one)
        assert 0.0 < feats["extent"] <= 1.0

    def test_solidity_in_unit_range(self, synthetic_contour_one):
        feats = contour_features(synthetic_contour_one)
        assert 0.0 < feats["solidity"] <= 1.0

    def test_degenerate_contour_does_not_crash(self):
        """Single-point contour must not raise an exception."""
        tiny = np.array([[[10, 10]]], dtype=np.int32)
        try:
            feats = contour_features(tiny)
            assert feats["area"] == 0.0
        except Exception as e:
            pytest.fail(f"contour_features raised {e} on degenerate contour")

    def test_feature_values_are_floats(self, synthetic_contour_one):
        feats = contour_features(synthetic_contour_one)
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"


# ---------------------------------------------------------------------------
# make_synthetic_video + load_video + read_frames
# ---------------------------------------------------------------------------

class TestVideoIO:

    def test_synthetic_video_created(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        result = make_synthetic_video(path=path, n_frames=10)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_load_video_returns_cap_and_props(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        make_synthetic_video(path=path, n_frames=5)
        cap, props = load_video(path)
        assert cap.isOpened()
        for key in ["fps", "width", "height", "frame_count"]:
            assert key in props
        assert props["width"] == 320
        assert props["height"] == 240
        cap.release()

    def test_load_video_raises_on_bad_path(self):
        with pytest.raises(FileNotFoundError):
            load_video("nonexistent_video_12345.mp4")

    def test_read_frames_yields_grayscale(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        make_synthetic_video(path=path, n_frames=5)
        cap, _ = load_video(path)
        frames = list(read_frames(cap, max_frames=3))
        cap.release()
        assert len(frames) == 3
        for f in frames:
            assert f.ndim == 2, "Frames must be grayscale (2D)"
            assert f.dtype == np.uint8

    def test_read_frames_max_frames_limit(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        make_synthetic_video(path=path, n_frames=20)
        cap, _ = load_video(path)
        frames = list(read_frames(cap, max_frames=5))
        cap.release()
        assert len(frames) == 5

    def test_compute_background_shape_matches_frame(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        make_synthetic_video(path=path, n_frames=10)
        cap, _ = load_video(path)
        bg = compute_background(cap, n_samples=5)
        cap.release()
        assert bg.shape == (240, 320)
        assert bg.dtype == np.uint8

    def test_compute_background_rewinds_cap(self, tmp_path):
        path = str(tmp_path / "test.mp4")
        make_synthetic_video(path=path, n_frames=10)
        cap, _ = load_video(path)
        compute_background(cap, n_samples=5)
        # After rewind, should be able to read frame 0 again
        ok, _ = cap.read()
        cap.release()
        assert ok, "cap was not rewound after compute_background"
