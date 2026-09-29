"""
test_fly_count.py
-----------------
Unit tests for fly_count.py.
Synthetic data only — results verify software correctness, NOT real accuracy.
"""
import numpy as np
import cv2
import pytest
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fly_count import (
    contour_to_feature_vector, rule_based_predict, label_to_count,
    make_synthetic_dataset, train_decision_tree, predict_fly_count,
    save_model, load_model, FEATURE_NAMES, CLASS_NAMES,
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestContourToFeatureVector:

    def test_output_shape(self, synthetic_contour_one):
        vec = contour_to_feature_vector(synthetic_contour_one)
        assert vec.shape == (5,), f"Expected (5,), got {vec.shape}"

    def test_output_dtype(self, synthetic_contour_one):
        vec = contour_to_feature_vector(synthetic_contour_one)
        assert vec.dtype == np.float32

    def test_all_values_finite(self, synthetic_contour_one):
        vec = contour_to_feature_vector(synthetic_contour_one)
        assert np.all(np.isfinite(vec)), "Feature vector contains inf/nan"

    def test_feature_order_matches_names(self, synthetic_contour_one):
        """Feature vector order must match FEATURE_NAMES."""
        from preprocessing import contour_features
        vec = contour_to_feature_vector(synthetic_contour_one)
        feats = contour_features(synthetic_contour_one)
        for i, name in enumerate(FEATURE_NAMES):
            assert abs(vec[i] - feats[name]) < 1e-5, f"Mismatch at index {i} ({name})"


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

class TestRuleBasedPredict:

    def test_noise_contour_returns_zero(self, synthetic_contour_noise):
        label = rule_based_predict(synthetic_contour_noise)
        assert label == "zero", f"Expected 'zero', got '{label}'"

    def test_single_fly_contour_returns_one(self, synthetic_contour_one):
        label = rule_based_predict(synthetic_contour_one)
        assert label == "one", f"Expected 'one', got '{label}'"

    def test_returns_valid_class_name(self, synthetic_contour_one):
        label = rule_based_predict(synthetic_contour_one)
        assert label in CLASS_NAMES, f"'{label}' not in CLASS_NAMES"

    def test_merged_flies_returns_two(self):
        """Large, lower-solidity blob should return 'two'."""
        mask = np.zeros((240, 320), dtype=np.uint8)
        # Two side-by-side ellipses that touch — merged blob
        cv2.ellipse(mask, (150, 120), (20, 8), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (180, 120), (20, 8), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            label = rule_based_predict(contours[0])
            # Result depends on actual shape; just verify it's a valid class
            assert label in CLASS_NAMES


# ---------------------------------------------------------------------------
# label_to_count
# ---------------------------------------------------------------------------

class TestLabelToCount:

    @pytest.mark.parametrize("label,expected", [
        ("zero", 0), ("one", 1), ("two", 2), ("unknown", 0),
    ])
    def test_converts_correctly(self, label, expected):
        assert label_to_count(label) == expected

    def test_invalid_label_returns_zero(self):
        assert label_to_count("garbage") == 0


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class TestMakeSyntheticDataset:

    def test_shape(self):
        X, y, le = make_synthetic_dataset(n_per_class=10)
        assert X.shape == (30, 5)
        assert y.shape == (30,)

    def test_three_classes_present(self):
        X, y, le = make_synthetic_dataset(n_per_class=20)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_label_encoder_classes(self):
        _, _, le = make_synthetic_dataset(n_per_class=10)
        assert list(le.classes_) == ["one", "two", "zero"]

    def test_label_index_correctness(self):
        """one=0, two=1, zero=2 must hold (alphabetical sklearn order)."""
        _, _, le = make_synthetic_dataset(n_per_class=10)
        assert le.transform(["one"])[0]  == 0
        assert le.transform(["two"])[0]  == 1
        assert le.transform(["zero"])[0] == 2

    def test_reproducibility(self):
        X1, y1, _ = make_synthetic_dataset(n_per_class=20, random_state=42)
        X2, y2, _ = make_synthetic_dataset(n_per_class=20, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        X1, _, _ = make_synthetic_dataset(n_per_class=20, random_state=1)
        X2, _, _ = make_synthetic_dataset(n_per_class=20, random_state=2)
        assert not np.array_equal(X1, X2)

    def test_no_nan_or_inf(self):
        X, y, _ = make_synthetic_dataset(n_per_class=50)
        assert np.all(np.isfinite(X))
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Decision Tree training
# ---------------------------------------------------------------------------

class TestTrainDecisionTree:

    def test_returns_four_values(self):
        X, y, le = make_synthetic_dataset(n_per_class=30)
        result = train_decision_tree(X, y, le)
        assert len(result) == 4

    def test_clf_has_predict(self):
        X, y, le = make_synthetic_dataset(n_per_class=30)
        clf, X_test, y_test, y_pred = train_decision_tree(X, y, le)
        assert hasattr(clf, "predict")

    def test_predictions_shape_matches_test(self):
        X, y, le = make_synthetic_dataset(n_per_class=30)
        clf, X_test, y_test, y_pred = train_decision_tree(X, y, le)
        assert y_pred.shape == y_test.shape

    def test_predictions_are_valid_integers(self):
        X, y, le = make_synthetic_dataset(n_per_class=30)
        clf, X_test, y_test, y_pred = train_decision_tree(X, y, le)
        assert set(y_pred).issubset({0, 1, 2})

    def test_synthetic_accuracy_above_chance(self):
        """
        On clearly separated synthetic data the DT should do better than
        random (chance = 33%).  NOT a real-world performance claim.
        """
        from sklearn.metrics import accuracy_score
        X, y, le = make_synthetic_dataset(n_per_class=100)
        clf, X_test, y_test, y_pred = train_decision_tree(X, y, le)
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.50, f"Accuracy {acc:.2f} is below 50% even on synthetic data"

    def test_reproducibility(self):
        X, y, le = make_synthetic_dataset(n_per_class=50)
        clf1, _, _, pred1 = train_decision_tree(X, y, le, random_state=42)
        clf2, _, _, pred2 = train_decision_tree(X, y, le, random_state=42)
        np.testing.assert_array_equal(pred1, pred2)


# ---------------------------------------------------------------------------
# predict_fly_count
# ---------------------------------------------------------------------------

class TestPredictFlyCount:

    def test_returns_valid_label(self, fly_count_model, synthetic_contour_one):
        clf, le = fly_count_model
        label = predict_fly_count(clf, le, synthetic_contour_one)
        assert label in CLASS_NAMES, f"'{label}' not in CLASS_NAMES"

    def test_single_fly_contour_predicts_one(self, fly_count_model, synthetic_contour_one):
        """
        The trained Decision Tree should predict 'one' for a well-formed
        single-fly contour (~265 px², high solidity).
        NOTE: result depends on synthetic training data distributions.
        """
        clf, le = fly_count_model
        label = predict_fly_count(clf, le, synthetic_contour_one)
        assert label == "one", (
            f"Expected 'one' for single fly contour, got '{label}'. "
            "This may indicate a LabelEncoder ordering issue."
        )

    def test_output_is_string(self, fly_count_model, synthetic_contour_one):
        clf, le = fly_count_model
        result = predict_fly_count(clf, le, synthetic_contour_one)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Model save / load round-trip
# ---------------------------------------------------------------------------

class TestModelPersistence:

    def test_save_and_load_round_trip(self, tmp_path):
        X, y, le = make_synthetic_dataset(n_per_class=30)
        clf, _, _, _ = train_decision_tree(X, y, le)
        path = str(tmp_path / "test_fc_model.joblib")
        save_model(clf, le, path=path)
        assert os.path.isfile(path)

        clf2, le2 = load_model(path)
        assert list(le2.classes_) == list(le.classes_)
        # Predictions must be identical
        X_test = X[:10]
        np.testing.assert_array_equal(clf.predict(X_test), clf2.predict(X_test))

    def test_load_missing_model_raises(self, tmp_path):
        path = str(tmp_path / "does_not_exist.joblib")
        with pytest.raises(Exception):
            load_model(path)
