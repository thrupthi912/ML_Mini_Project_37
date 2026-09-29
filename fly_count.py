"""
fly_count.py  — Thrupthi (Person 1, Stage 1)
---------------------------------------------
Decision Tree classifier that labels each contour blob as containing:
  "zero"  — background noise / not a fly
  "one"   — a single fly
  "two"   — two flies merged into one blob

Algorithm overview
------------------
1. Extract geometric features from each contour:
   area, perimeter, aspect_ratio, extent, solidity
2. If labeled data (input/images/fly_count_labels.csv) is present, load it
   and train a DecisionTreeClassifier.
3. If no real data is available, a synthetic dataset is generated for a
   full dry-run so the code always runs and can be demonstrated.
4. Evaluate with accuracy, classification report, and confusion matrix.
5. Save the trained model to models/fly_count_model.joblib.
6. Export a text representation of the decision tree.

Usage
-----
    python fly_count.py              # auto-uses synthetic data if real not found
    python fly_count.py --data input/images/fly_count_labels.csv
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# Local module
from preprocessing import (make_synthetic_video, load_video, compute_background,
                            read_frames, threshold_frame, extract_contours,
                            contour_features)


# ---------------------------------------------------------------------------
# Feature / label helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["area", "perimeter", "aspect_ratio", "extent", "solidity"]
CLASS_NAMES   = ["zero", "one", "two"]


def contour_to_feature_vector(contour) -> np.ndarray:
    """Convert a single contour into a 1-D feature array."""
    feats = contour_features(contour)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Load real labeled data
# ---------------------------------------------------------------------------

def load_labeled_data(csv_path: str):
    """
    Load a CSV file with columns:
        area, perimeter, aspect_ratio, extent, solidity, label

    where label is one of: zero, one, two

    Returns
    -------
    X : np.ndarray  shape (N, 5)
    y : np.ndarray  shape (N,)  — integer-encoded labels
    le : LabelEncoder
    """
    import csv
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    X = np.array([[float(r[name]) for name in FEATURE_NAMES]
                  for r in rows], dtype=np.float32)
    raw_labels = [r["label"].strip() for r in rows]

    le = LabelEncoder()
    le.fit(CLASS_NAMES)          # fix class ordering
    y = le.transform(raw_labels)
    return X, y, le


# ---------------------------------------------------------------------------
# Synthetic dataset generation (fallback when no real data exists)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n_per_class: int = 120, random_state: int = 42):
    """
    Generate a plausible synthetic feature dataset for the three classes.

    The distributions are based on typical contour measurements observed
    in fruit-fly arena videos (arena ~320x240 pixels, flies ~10-14 px body).

    Zero flies  — small, irregular noise blobs
    One fly     — medium ellipse-shaped blob
    Two flies   — larger, less compact blob (two bodies touching)

    Returns
    -------
    X : np.ndarray  (3*n_per_class, 5)
    y : np.ndarray  (3*n_per_class,)   integer labels
    le : LabelEncoder
    """
    rng = np.random.default_rng(random_state)

    def sample(n, area_mu, area_sig,
               perim_mu, perim_sig,
               ar_mu, ar_sig,
               ext_mu, ext_sig,
               sol_mu, sol_sig):
        return np.column_stack([
            rng.normal(area_mu,  area_sig,  n).clip(10, 6000),
            rng.normal(perim_mu, perim_sig, n).clip(10, 500),
            rng.normal(ar_mu,    ar_sig,    n).clip(0.2, 4.0),
            rng.normal(ext_mu,   ext_sig,   n).clip(0.05, 1.0),
            rng.normal(sol_mu,   sol_sig,   n).clip(0.3, 1.0),
        ]).astype(np.float32)

    # zero: tiny noise blobs
    X_zero = sample(n_per_class,
                    area_mu=40,   area_sig=15,
                    perim_mu=30,  perim_sig=10,
                    ar_mu=1.1,    ar_sig=0.4,
                    ext_mu=0.55,  ext_sig=0.12,
                    sol_mu=0.70,  sol_sig=0.10)

    # one fly: matches typical contour areas from synthetic video (~265 px)
    X_one  = sample(n_per_class,
                    area_mu=265,  area_sig=40,
                    perim_mu=75,  perim_sig=12,
                    ar_mu=1.67,   ar_sig=0.3,
                    ext_mu=0.62,  ext_sig=0.08,
                    sol_mu=0.97,  sol_sig=0.02)

    # two flies merged: roughly 2x single fly area, lower solidity
    X_two  = sample(n_per_class,
                    area_mu=530,  area_sig=80,
                    perim_mu=140, perim_sig=22,
                    ar_mu=2.5,    ar_sig=0.5,
                    ext_mu=0.50,  ext_sig=0.10,
                    sol_mu=0.72,  sol_sig=0.08)

    X = np.vstack([X_zero, X_one, X_two])

    le = LabelEncoder()
    le.fit(CLASS_NAMES)   # sorts alphabetically: one=0, two=1, zero=2

    # Use le.transform so indices always match the encoder's class ordering
    labels_str = (["zero"] * n_per_class +
                  ["one"]  * n_per_class +
                  ["two"]  * n_per_class)
    y = le.transform(labels_str)

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx], le


# ---------------------------------------------------------------------------
# Build dataset from a real video (used when video + manual labels available)
# ---------------------------------------------------------------------------

def build_dataset_from_video(video_path: str,
                              label_map: dict = None,
                              max_frames: int = 200):
    """
    Process a video and collect (feature_vector, label) pairs.

    If label_map is None, every contour is labeled "one" by default — this is
    only useful as a structural test.  For real training you need ground-truth
    labels (see README for the CSV format).

    Parameters
    ----------
    video_path : str
    label_map  : dict  {frame_idx: {contour_idx: label_str}}  or None
    max_frames : int

    Returns
    -------
    X : np.ndarray
    y : list of str (raw labels)
    """
    cap, props = load_video(video_path)
    background = compute_background(cap)

    X_rows, y_rows = [], []

    for frame_idx, frame in enumerate(read_frames(cap, max_frames=max_frames)):
        mask = threshold_frame(frame, background)
        contours = extract_contours(mask)

        for c_idx, contour in enumerate(contours):
            feat_vec = contour_to_feature_vector(contour)

            if label_map and frame_idx in label_map:
                label = label_map[frame_idx].get(c_idx, "one")
            else:
                # Heuristic labeling based on area thresholds (rough guess)
                area = feat_vec[0]
                if area < 80:
                    label = "zero"
                elif area < 420:
                    label = "one"
                else:
                    label = "two"

            X_rows.append(feat_vec)
            y_rows.append(label)

    cap.release()
    return np.array(X_rows, dtype=np.float32), y_rows


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_fly_count_classifier(X: np.ndarray,
                                y: np.ndarray,
                                le: LabelEncoder,
                                max_depth: int = 5,
                                random_state: int = 42):
    """
    Split data, fit a DecisionTreeClassifier, and return the trained model
    alongside test-set predictions.

    Parameters
    ----------
    X            : feature matrix  (N, 5)
    y            : integer label array  (N,)
    le           : fitted LabelEncoder (maps int ↔ class name)
    max_depth    : max depth of the decision tree (controls complexity)
    random_state : reproducibility seed

    Returns
    -------
    clf     : trained DecisionTreeClassifier
    X_test  : test features
    y_test  : true test labels (int)
    y_pred  : predicted test labels (int)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=5,      # avoid tiny leaves = overfitting
        class_weight="balanced", # handles any class imbalance
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# Evaluation & visualisation
# ---------------------------------------------------------------------------

def evaluate(clf, X_test, y_test, y_pred, le, save_dir: str = "output"):
    """
    Print and save evaluation metrics:
      - Accuracy
      - Full classification report
      - Confusion matrix PNG
      - Decision tree diagram PNG
      - Text representation of the tree
    """
    os.makedirs(save_dir, exist_ok=True)
    class_labels = le.classes_

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=class_labels))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_labels)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Fly Count — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_fly_count.png")
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix → {cm_path}")

    # --- Decision tree diagram ---
    fig2, ax2 = plt.subplots(figsize=(16, 7))
    plot_tree(clf,
              feature_names=FEATURE_NAMES,
              class_names=list(class_labels),
              filled=True,
              rounded=True,
              ax=ax2,
              fontsize=8)
    ax2.set_title("Decision Tree — Fly Count Classifier")
    tree_img_path = os.path.join(save_dir, "tree_fly_count.png")
    plt.savefig(tree_img_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved decision tree diagram → {tree_img_path}")

    # --- Text tree ---
    tree_text = export_text(clf, feature_names=FEATURE_NAMES)
    txt_path = os.path.join(save_dir, "tree_fly_count.txt")
    with open(txt_path, "w") as f:
        f.write(tree_text)
    print(f"  Saved decision tree text → {txt_path}")

    return acc


# ---------------------------------------------------------------------------
# Save / load model
# ---------------------------------------------------------------------------

def save_model(clf, le, path: str = "models/fly_count_model.joblib"):
    """Save the classifier and label encoder together."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, path)
    print(f"  Model saved → {path}")


def load_model(path: str = "models/fly_count_model.joblib"):
    """Load a previously saved fly-count model."""
    bundle = joblib.load(path)
    return bundle["clf"], bundle["le"]


# ---------------------------------------------------------------------------
# Prediction helper (used by main.py)
# ---------------------------------------------------------------------------

def predict_fly_count(clf, le, contour) -> str:
    """
    Predict the fly count label for a single contour.

    Parameters
    ----------
    clf     : trained DecisionTreeClassifier
    le      : fitted LabelEncoder
    contour : OpenCV contour array

    Returns
    -------
    label : str — "zero", "one", or "two"
    """
    feat_vec = contour_to_feature_vector(contour).reshape(1, -1)
    int_label = clf.predict(feat_vec)[0]
    return le.inverse_transform([int_label])[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_csv: str = None, save_dir: str = "output"):
    print("=" * 55)
    print("  FlyCount Decision Tree Classifier — Thrupthi")
    print("=" * 55)

    # ---- Data loading -------------------------------------------------------
    if data_csv and os.path.exists(data_csv):
        print(f"\n[DATA] Loading real labeled data from: {data_csv}")
        X, y, le = load_labeled_data(data_csv)
        print(f"       Loaded {len(y)} samples. "
              f"Class distribution: { {le.classes_[i]: int((y==i).sum()) for i in range(len(le.classes_))} }")
    else:
        if data_csv:
            print(f"\n[WARN] CSV not found: {data_csv}")
        print("\n[DATA] No labeled CSV found — using SYNTHETIC dataset for demo.")
        print("       To use real data, provide --data path/to/fly_count_labels.csv")
        print("       (see README.md for the expected CSV format)\n")
        X, y, le = make_synthetic_dataset(n_per_class=150)
        print(f"       Generated {len(y)} synthetic samples across 3 classes.")

    # ---- Training -----------------------------------------------------------
    print("\n[TRAIN] Fitting Decision Tree (max_depth=5) ...")
    clf, X_test, y_test, y_pred = train_fly_count_classifier(X, y, le)
    print("        Training complete.")

    # ---- Evaluation ---------------------------------------------------------
    print("\n[EVAL]  Evaluating on held-out test set (25%) ...")
    acc = evaluate(clf, X_test, y_test, y_pred, le, save_dir=save_dir)

    # ---- Save ---------------------------------------------------------------
    print("\n[SAVE]  Saving model ...")
    save_model(clf, le)

    print("\n[DONE]  Fly count module finished.")
    print(f"        Test accuracy: {acc*100:.2f}%")
    if data_csv is None or not os.path.exists(data_csv):
        print("\n[NOTE]  Results above are on SYNTHETIC data.")
        print("        Accuracy will differ (likely lower) on real video data.")
        print("        Download the labeled dataset — see README.md for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the FlyCount Decision Tree classifier")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to fly_count_labels.csv "
                             "(omit to run on synthetic demo data)")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save outputs (default: output/)")
    args = parser.parse_args()

    main(data_csv=args.data, save_dir=args.save_dir)
