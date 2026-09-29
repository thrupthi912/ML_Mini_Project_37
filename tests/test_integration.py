"""
test_integration.py
-------------------
Integration tests for the full main.py pipeline.
Uses synthetic video only — no real dataset required.
Results verify pipeline connectivity and error handling, NOT accuracy.
"""
import numpy as np
import cv2
import pytest
import os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing import make_synthetic_video, load_video, compute_background, read_frames
from main import load_all_models, process_frame, annotate_frame, _module_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_video_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "synth.mp4")
    make_synthetic_video(path=path, n_frames=15)
    return path


@pytest.fixture(scope="module")
def pipeline_components(synthetic_video_path):
    """Load models + background once for all integration tests."""
    models = load_all_models()
    cap, props = load_video(synthetic_video_path)
    bg = compute_background(cap, n_samples=10)
    frame = next(read_frames(cap, max_frames=1))
    cap.release()
    return models, bg, frame


# ---------------------------------------------------------------------------
# load_all_models
# ---------------------------------------------------------------------------

class TestLoadAllModels:

    def test_returns_dict_with_required_keys(self):
        models = load_all_models()
        for key in ("fly_count", "orientation", "sex", "wing"):
            assert key in models, f"Missing key: {key}"

    def test_each_entry_has_ok_flag(self):
        models = load_all_models()
        for key, val in models.items():
            assert "ok" in val, f"Entry '{key}' missing 'ok' flag"
            assert isinstance(val["ok"], bool)

    def test_fly_count_loaded(self):
        """FlyCount model must load since we have fly_count_model.joblib."""
        models = load_all_models()
        assert models["fly_count"]["ok"], (
            "FlyCount model not loaded. Run: python fly_count.py first."
        )

    def test_missing_model_ok_is_false(self):
        """Sex and orientation models are absent — ok must be False."""
        models = load_all_models()
        assert models["sex"]["ok"] is False or models["sex"]["model"] is None


# ---------------------------------------------------------------------------
# _module_summary
# ---------------------------------------------------------------------------

class TestModuleSummary:

    def test_returns_string(self):
        models = load_all_models()
        summary = _module_summary(models)
        assert isinstance(summary, str)

    def test_contains_all_modules(self):
        models = load_all_models()
        summary = _module_summary(models)
        for token in ("FlyCount", "Orient", "Sex", "Wing"):
            assert token in summary, f"'{token}' missing from summary: {summary}"


# ---------------------------------------------------------------------------
# process_frame
# ---------------------------------------------------------------------------

class TestProcessFrame:

    def test_returns_dict_with_required_keys(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        for key in ("contours", "labels", "counts", "orientations",
                    "sexes", "wing_info", "total_flies", "errors"):
            assert key in result, f"Missing key: {key}"

    def test_labels_are_valid_class_names(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        valid = {"zero", "one", "two"}
        for lbl in result["labels"]:
            assert lbl in valid, f"Invalid label: '{lbl}'"

    def test_counts_match_labels(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        for lbl, cnt in zip(result["labels"], result["counts"]):
            expected = {"zero": 0, "one": 1, "two": 2}[lbl]
            assert cnt == expected

    def test_total_flies_is_sum_of_counts(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        assert result["total_flies"] == sum(result["counts"])

    def test_list_lengths_consistent(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        n = len(result["contours"])
        assert len(result["labels"])       == n
        assert len(result["counts"])       == n
        assert len(result["orientations"]) == n
        assert len(result["sexes"])        == n

    def test_errors_is_list(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        assert isinstance(result["errors"], list)

    def test_detects_two_flies_in_synthetic_frame(self, pipeline_components):
        """
        The synthetic frame has exactly 2 fly blobs — total_flies must be 2.
        """
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        assert result["total_flies"] == 2, (
            f"Expected 2 flies in synthetic frame, got {result['total_flies']}"
        )

    def test_none_background_does_not_crash(self, pipeline_components):
        """Pipeline must not crash when background is None."""
        models, _, frame = pipeline_components
        result = process_frame(frame, None, models)
        assert "total_flies" in result

    def test_wing_info_absent_when_model_missing(self, pipeline_components):
        """If wing model is not loaded, wing_info must remain empty."""
        models, bg, frame = pipeline_components
        if models["wing"]["ok"]:
            pytest.skip("Wing model is present — test only applies when absent.")
        result = process_frame(frame, bg, models)
        assert result["wing_info"] == {}

    def test_sexes_are_none_when_model_missing(self, pipeline_components):
        """If sex model is absent, all sex entries must be None."""
        models, bg, frame = pipeline_components
        if models["sex"]["ok"]:
            pytest.skip("Sex model present — test only applies when absent.")
        result = process_frame(frame, bg, models)
        for s in result["sexes"]:
            assert s is None


# ---------------------------------------------------------------------------
# annotate_frame
# ---------------------------------------------------------------------------

class TestAnnotateFrame:

    def test_output_is_bgr(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        annotated = annotate_frame(frame, result, models, fps=30.0)
        assert annotated.ndim == 3
        assert annotated.shape[2] == 3

    def test_output_shape_matches_input(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        annotated = annotate_frame(frame, result, models, fps=30.0)
        assert annotated.shape[:2] == frame.shape[:2]

    def test_output_dtype_uint8(self, pipeline_components):
        models, bg, frame = pipeline_components
        result = process_frame(frame, bg, models)
        annotated = annotate_frame(frame, result, models, fps=30.0)
        assert annotated.dtype == np.uint8

    def test_empty_result_does_not_crash(self, pipeline_components):
        """Empty contour list must not raise an exception."""
        models, bg, frame = pipeline_components
        empty_result = {
            "contours": [], "labels": [], "counts": [],
            "orientations": [], "sexes": [], "wing_info": {},
            "total_flies": 0, "errors": [],
        }
        annotated = annotate_frame(frame, empty_result, models, fps=0.0)
        assert annotated is not None


# ---------------------------------------------------------------------------
# FPS measurement
# ---------------------------------------------------------------------------

class TestFPS:

    def test_processing_time_measured(self, pipeline_components):
        """Verify we can measure per-frame time — basic sanity check."""
        models, bg, frame = pipeline_components
        t0 = time.perf_counter()
        process_frame(frame, bg, models)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        assert elapsed > 0
        fps = 1.0 / elapsed
        # On any modern machine processing one 320x240 frame should be > 1 FPS
        assert fps > 1.0, f"FPS unexpectedly low: {fps:.2f}"
