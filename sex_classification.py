"""
sex_classification.py
---------------------
Male / female identity classification for fruit fly video analysis.

Algorithm
---------
Features (per contour)
  1. area          — female Drosophila have a larger abdomen; their body
                     contour area is typically 10-25% greater than the male.
  2. aspect_ratio  — the female body is more rounded (lower AR); the male
                     is more slender (higher AR due to the narrow abdomen).
  3. perimeter     — correlated with size but adds independent shape info.
  4. extent        — area / bounding-box area; females are more compact.
  5. solidity      — area / convex-hull area; robust to minor pose changes.

All features are standardized with StandardScaler (zero mean, unit variance)
before being passed to the classifier.  This is required for Logistic
Regression because it is sensitive to feature scale.

Classifier
  LogisticRegression (L2 penalty, solver=lbfgs).
  A soft-margin linear boundary in the standardized feature space.
  Returns both a hard label ("male" / "female") and a probability.

Data policy
-----------
This module performs a strict check for labeled training data before
doing anything else.  If the CSV file is absent or empty:
  - NO model is trained.
  - NO synthetic labels are fabricated.
  - classify_pair() returns None and logs a clear reason.
  - The module exits with a plain explanation of what data is needed.

Expected CSV format  (input/images/sex_labels.csv)
  Columns: area, perimeter, aspect_ratio, extent, solidity, label
  Labels : "male"  or  "female"
  One row per labeled contour patch.

Usage
-----
    python sex_classification.py                  # data check + status report
    python sex_classification.py --data input/images/sex_labels.csv
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)

from preprocessing import contour_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES  = ["area", "perimeter", "aspect_ratio", "extent", "solidity"]
CLASS_NAMES    = ["female", "male"]          # alphabetical = scikit-learn default
MODEL_PATH     = "models/sex_clf_model.joblib"
DEFAULT_CSV    = "input/images/sex_labels.csv"

# Colors for visualization (BGR for OpenCV, RGB for matplotlib)
COLORS_BGR = {"male": (255, 100, 0), "female": (0, 100, 255), "unknown": (180, 180, 180)}
COLORS_RGB = {"male": (1.0, 0.39, 0.0), "female": (0.0, 0.39, 1.0), "unknown": (0.7, 0.7, 0.7)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(contour) -> np.ndarray:
    """
    Compute the 5-element feature vector for a single fly contour.

    Features
    --------
    area         : contour pixel area
    perimeter    : arc length
    aspect_ratio : bounding-rect width / height
    extent       : area / bounding-rect area
    solidity     : area / convex-hull area

    Returns
    -------
    np.ndarray  shape (5,)  dtype float32
    """
    feats = contour_features(contour)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# DATA CHECK  — strict, no fabrication
# ---------------------------------------------------------------------------

def check_for_labeled_data(csv_path: str) -> bool:
    """
    Verify that a usable labeled CSV exists.

    Returns True only if:
      - the file exists
      - it has at least one header row + one data row
      - it contains both "male" and "female" labels (need both classes to train)

    Prints a detailed status message in all cases.
    Does NOT raise exceptions — caller decides what to do with the result.
    """
    print("[DATA CHECK] Looking for labeled sex data ...")
    print(f"             Path : {csv_path}")

    if not os.path.isfile(csv_path):
        print("[DATA CHECK] NOT FOUND.")
        _print_data_instructions(csv_path)
        return False

    try:
        import csv
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"[DATA CHECK] Could not read file: {e}")
        _print_data_instructions(csv_path)
        return False

    if len(rows) == 0:
        print("[DATA CHECK] File exists but is EMPTY (no data rows).")
        _print_data_instructions(csv_path)
        return False

    # Check required columns
    required = set(FEATURE_NAMES) | {"label"}
    missing_cols = required - set(rows[0].keys())
    if missing_cols:
        print(f"[DATA CHECK] Missing columns: {missing_cols}")
        _print_data_instructions(csv_path)
        return False

    labels_present = {r["label"].strip().lower() for r in rows}
    if not {"male", "female"}.issubset(labels_present):
        print(f"[DATA CHECK] Both 'male' and 'female' labels needed. "
              f"Found: {labels_present}")
        _print_data_instructions(csv_path)
        return False

    male_count   = sum(1 for r in rows if r["label"].strip().lower() == "male")
    female_count = sum(1 for r in rows if r["label"].strip().lower() == "female")
    print(f"[DATA CHECK] FOUND  —  {len(rows)} labeled rows  "
          f"(male: {male_count}, female: {female_count})")
    return True


def _print_data_instructions(csv_path: str):
    """Print a clear explanation of what labeled data is needed."""
    print()
    print("  No training will be performed without real labeled data.")
    print("  No synthetic labels will be fabricated.")
    print()
    print("  To train this module you need:")
    print(f"    {csv_path}")
    print()
    print("  CSV format:")
    print("    area,perimeter,aspect_ratio,extent,solidity,label")
    print("    310.5,88.2,1.82,0.61,0.94,male")
    print("    385.0,95.1,1.51,0.68,0.93,female")
    print("    ...")
    print()
    print("  How to collect labels:")
    print("    1. Download the CS229 labeled dataset (see README.md).")
    print("    2. Run contour extraction on labeled frames.")
    print("    3. Match each contour to its ground-truth sex annotation.")
    print("    4. Compute the 5 features and write one row per contour.")
    print()
    print("  classify_pair() will return None until a trained model exists.")


# ---------------------------------------------------------------------------
# LOAD LABELED CSV
# ---------------------------------------------------------------------------

def load_labeled_data(csv_path: str):
    """
    Load the labeled CSV and return feature matrix + label array.

    Returns
    -------
    X      : np.ndarray  (N, 5)  float32
    y      : np.ndarray  (N,)    int  (0=female, 1=male — sklearn alphabetical)
    labels : list of str         raw label strings
    """
    import csv
    from sklearn.preprocessing import LabelEncoder

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X = np.array(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    raw_labels = [row["label"].strip().lower() for row in rows]

    le = LabelEncoder()
    le.fit(CLASS_NAMES)          # fixes ordering: female=0, male=1
    y  = le.transform(raw_labels)

    return X, y, raw_labels


# ---------------------------------------------------------------------------
# TRAIN  — Logistic Regression with StandardScaler
# ---------------------------------------------------------------------------

def train_classifier(X: np.ndarray,
                     y: np.ndarray,
                     random_state: int = 42):
    """
    Build and fit a StandardScaler → LogisticRegression pipeline.

    Why StandardScaler?
      Logistic Regression optimizes a loss function via gradient descent.
      Features on very different scales (area ~300, aspect_ratio ~1.6)
      cause poorly conditioned gradients, slow convergence, and biased
      coefficients.  Standardizing to zero mean / unit variance fixes this.

    Why LogisticRegression?
      - Produces calibrated probabilities (useful for the pair-ranking step).
      - Interpretable: the sign of each coefficient shows which sex has
        higher area / AR / etc.
      - Fast to train; regularized (L2) to prevent overfitting on small datasets.

    Returns
    -------
    model   : fitted Pipeline(StandardScaler, LogisticRegression)
    X_test  : held-out feature matrix  (25%)
    y_test  : held-out true labels
    y_pred  : held-out predictions
    cv_scores : 5-fold cross-validation accuracy array
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",   # robust to any class imbalance
            random_state=random_state,
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 5-fold cross-validation on the full training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    return model, X_test, y_test, y_pred, cv_scores


# ---------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, y_pred, cv_scores,
             save_dir: str = "output"):
    """
    Print metrics and save:
      - Confusion matrix PNG
      - Coefficient bar chart PNG  (which features drive the decision)
    """
    os.makedirs(save_dir, exist_ok=True)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Hold-out accuracy  : {acc*100:.2f}%")
    print(f"  5-fold CV accuracy : {cv_scores.mean()*100:.2f}%  "
          f"(± {cv_scores.std()*100:.2f}%)")
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # -- Confusion matrix --------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    ax.set_title("Sex Classification — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_sex.png")
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix  → {cm_path}")

    # -- Coefficient bar chart  --------------------------------------------
    # The scaler+LR pipeline: coefficients are in the standardized space.
    lr   = model.named_steps["classifier"]
    coef = lr.coef_[0]            # shape (n_features,) for binary

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#e05c5c" if c > 0 else "#5c7de0" for c in coef]
    ax.barh(FEATURE_NAMES, coef, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression coefficient\n"
                  "(positive → predicts male,  negative → predicts female)")
    ax.set_title("Sex Classifier — Feature Importance (standardized)")
    plt.tight_layout()
    coef_path = os.path.join(save_dir, "sex_clf_coefficients.png")
    plt.savefig(coef_path, dpi=120)
    plt.close(fig)
    print(f"  Saved coefficient plot  → {coef_path}")

    return acc


# ---------------------------------------------------------------------------
# SAVE / LOAD MODEL
# ---------------------------------------------------------------------------

def save_model(model, path: str = MODEL_PATH):
    """Save the fitted Pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved → {path}")


def load_model(path: str = MODEL_PATH):
    """
    Load a trained sex classification model.
    Returns None (silently) if the file does not exist.
    """
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# SINGLE-CONTOUR PREDICTION
# ---------------------------------------------------------------------------

def predict_sex(model, contour) -> tuple:
    """
    Predict the sex of a single fly contour.

    Parameters
    ----------
    model   : fitted Pipeline  (StandardScaler → LogisticRegression)
    contour : OpenCV contour array

    Returns
    -------
    label : str    — "male" or "female"
    prob  : float  — probability of the predicted class  [0.5, 1.0]
    """
    feat_vec = extract_features(contour).reshape(1, -1)
    label    = model.predict(feat_vec)[0]              # int (0 or 1)
    prob     = float(model.predict_proba(feat_vec)[0, label])
    sex_str  = CLASS_NAMES[label]
    return sex_str, prob


# ---------------------------------------------------------------------------
# PAIR CLASSIFICATION  — the clean interface required by the task
# ---------------------------------------------------------------------------

def classify_pair(contour_a, contour_b, model=None):
    """
    Given two contours (each containing exactly one fly), predict which is
    the male and which is the female.

    The function uses the difference in model probability as a confidence
    measure rather than just the raw label, so the assignment is
    self-consistent: if contour_a is more likely male than contour_b, the
    pair assignment always reflects that.

    Parameters
    ----------
    contour_a : OpenCV contour  (first fly blob)
    contour_b : OpenCV contour  (second fly blob)
    model     : fitted Pipeline returned by train_classifier() / load_model()
                Pass None (or omit) to get an explicit "no model" response.

    Returns
    -------
    dict with keys:
        "male"    : the contour predicted to be the male
        "female"  : the contour predicted to be the female
        "male_prob"   : float  probability that the male contour is male
        "female_prob" : float  probability that the female contour is female
        "ordering"    : str    "AB" if a=male/b=female, "BA" if a=female/b=male

    Returns None if model is None, with a printed explanation.

    Example
    -------
    >>> result = classify_pair(c1, c2, model=loaded_model)
    >>> if result:
    ...     male_contour   = result["male"]
    ...     female_contour = result["female"]
    """
    if model is None:
        print("[classify_pair] No trained model available.")
        print("                Train first: python sex_classification.py --data <csv>")
        return None

    feat_a = extract_features(contour_a).reshape(1, -1)
    feat_b = extract_features(contour_b).reshape(1, -1)

    # Index 1 = "male" in the label encoder (alphabetical: female=0, male=1)
    male_idx = CLASS_NAMES.index("male")

    prob_a_male = float(model.predict_proba(feat_a)[0, male_idx])
    prob_b_male = float(model.predict_proba(feat_b)[0, male_idx])

    # Assign the higher-probability-male contour as male
    if prob_a_male >= prob_b_male:
        male_contour,   male_prob   = contour_a, prob_a_male
        female_contour, female_prob = contour_b, 1.0 - prob_b_male
        ordering = "AB"
    else:
        male_contour,   male_prob   = contour_b, prob_b_male
        female_contour, female_prob = contour_a, 1.0 - prob_a_male
        ordering = "BA"

    return {
        "male":        male_contour,
        "female":      female_contour,
        "male_prob":   male_prob,
        "female_prob": female_prob,
        "ordering":    ordering,
    }


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def visualize_pair_prediction(frame: np.ndarray,
                               contour_a,
                               contour_b,
                               result: dict,
                               save_path: str = None):
    """
    Draw both contours on the frame, colour-coded by predicted sex.

    Male   → orange  outline + "M" label
    Female → blue    outline + "F" label

    Saves a figure with the raw frame alongside the annotated frame.
    """
    if result is None:
        print("[visualize_pair_prediction] No result to draw (model not available).")
        return

    annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for sex_key, label_char in [("male", "M"), ("female", "F")]:
        contour = result[sex_key]
        prob    = result[f"{sex_key}_prob"]
        color   = COLORS_BGR[sex_key]

        cv2.drawContours(annotated, [contour], -1, color, 2)

        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

        text = f"{label_char} {prob*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(annotated,
                      (cx - 2, cy - th - 14),
                      (cx + tw + 2, cy - 10),
                      (0, 0, 0), -1)
        cv2.putText(annotated, text, (cx, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Grayscale Frame", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Sex Prediction  (ordering: {result['ordering']})\n"
        f"Male conf: {result['male_prob']*100:.1f}%   "
        f"Female conf: {result['female_prob']*100:.1f}%",
        fontsize=9
    )
    axes[1].axis("off")

    from matplotlib.patches import Patch
    legend = [
        Patch(color=COLORS_RGB["male"],   label="Male"),
        Patch(color=COLORS_RGB["female"], label="Female"),
    ]
    axes[1].legend(handles=legend, loc="lower right", fontsize=8)

    fig.suptitle("Sex Classification — Pair Prediction", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved visualization → {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# DEMO  — show what classify_pair does without a model
# ---------------------------------------------------------------------------

def _demo_no_model():
    """
    Run a self-contained demonstration of classify_pair()
    when no labeled data and no trained model exists.

    Creates two synthetic contours (ellipses) and shows that
    classify_pair() correctly refuses to predict without a model.
    """
    print()
    print("=" * 58)
    print("  DEMO: classify_pair() behaviour without a trained model")
    print("=" * 58)

    # Synthetic contours: small tight ellipse (male-like) and larger one
    dummy_frame  = np.full((200, 300), 220, dtype=np.uint8)
    mask_a = np.zeros_like(dummy_frame)
    mask_b = np.zeros_like(dummy_frame)
    cv2.ellipse(mask_a, (80, 100),  (12, 7),  0, 0, 360, 255, -1)   # small
    cv2.ellipse(mask_b, (220, 100), (15, 10), 0, 0, 360, 255, -1)   # larger

    contours_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    c_a = contours_a[0]
    c_b = contours_b[0]

    feats_a = extract_features(c_a)
    feats_b = extract_features(c_b)

    print(f"\n  Contour A features: "
          f"area={feats_a[0]:.1f}  AR={feats_a[2]:.2f}  "
          f"solidity={feats_a[4]:.2f}")
    print(f"  Contour B features: "
          f"area={feats_b[0]:.1f}  AR={feats_b[2]:.2f}  "
          f"solidity={feats_b[4]:.2f}")

    print("\n  Calling classify_pair(c_a, c_b, model=None):")
    result = classify_pair(c_a, c_b, model=None)
    print(f"  Return value: {result}")

    print()
    print("  This is the correct behaviour.")
    print("  Provide --data with a labeled CSV to train a real model.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(csv_path: str = DEFAULT_CSV, save_dir: str = "output"):
    print("=" * 58)
    print("  Sex Classification Module")
    print("=" * 58)
    print()

    # ── Step 1: strict data check ────────────────────────────────────────────
    has_data = check_for_labeled_data(csv_path)

    if not has_data:
        # Do NOT fabricate data. Show demo and exit cleanly.
        _demo_no_model()
        print()
        print("[STATUS] Module ready. Waiting for labeled training data.")
        print(f"         classify_pair() is implemented and will work once")
        print(f"         a model is trained and saved to: {MODEL_PATH}")
        return

    # ── Step 2: load + train ─────────────────────────────────────────────────
    print()
    print("[LOAD]  Reading labeled data ...")
    X, y, raw_labels = load_labeled_data(csv_path)
    print(f"        {len(y)} samples  |  "
          f"male: {(np.array(raw_labels)=='male').sum()}  "
          f"female: {(np.array(raw_labels)=='female').sum()}")

    print()
    print("[TRAIN] Fitting StandardScaler + LogisticRegression ...")
    model, X_test, y_test, y_pred, cv_scores = train_classifier(X, y)
    print("        Training complete.")

    # ── Step 3: evaluate ─────────────────────────────────────────────────────
    print()
    print("[EVAL]  Evaluating on held-out test set (25%) ...")
    acc = evaluate(model, X_test, y_test, y_pred, cv_scores, save_dir)

    # ── Step 4: save ─────────────────────────────────────────────────────────
    print()
    print("[SAVE]  Saving model ...")
    save_model(model)

    print()
    print(f"[DONE]  Sex classification module trained.  "
          f"Hold-out accuracy: {acc*100:.2f}%")
    print(f"        classify_pair() is now operational.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sex classification: Logistic Regression on contour features")
    parser.add_argument("--data", type=str, default=DEFAULT_CSV,
                        help=f"Path to labeled CSV  (default: {DEFAULT_CSV})")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save outputs  (default: output/)")
    args = parser.parse_args()

    main(csv_path=args.data, save_dir=args.save_dir)
