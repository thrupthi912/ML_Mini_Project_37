"""
fly_count.py  — Member 1, Stage 1
-----------------------------------
FlyCount module: classifies each contour blob as containing
  "zero"  — background noise / not a fly
  "one"   — a single fly
  "two"   — two flies merged into one blob

Data check at startup
---------------------
The module first checks for labeled training data at:
    input/images/fly_count_labels.csv

  FOUND  → trains a DecisionTreeClassifier on real labels.
  MISSING → falls back to a rule-based baseline (area thresholds).
            The baseline is clearly flagged in all output and has
            documented limitations (see rule_based_predict below).

Pipeline
--------
1. Grayscale frame  →  background subtraction
2. Gaussian blur + Otsu threshold  →  binary mask
3. cv2.findContours  →  contour list
4. Per contour: extract area, perimeter, aspect_ratio, extent, solidity
5. Decision Tree (or rule-based baseline)  →  label per contour
6. Sum labels  →  total fly count for the frame
7. Visualize: frame + mask + annotated contours with predicted labels

Usage
-----
    python fly_count.py                   # auto-detects data
    python fly_count.py --data input/images/fly_count_labels.csv
    python fly_count.py --video input/video/test4.mp4
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import joblib

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

from preprocessing import (make_synthetic_video, load_video, compute_background,
                            read_frames, threshold_frame, extract_contours,
                            contour_features)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["area", "perimeter", "aspect_ratio", "extent", "solidity"]
CLASS_NAMES   = ["zero", "one", "two"]

# Color map for drawing contours (BGR for OpenCV)
LABEL_COLORS_BGR = {
    "zero": (160, 160, 160),   # grey  — noise
    "one":  (0,   200,   0),   # green — single fly
    "two":  (0,   100, 255),   # orange — two flies merged
}
# Same colors in RGB for matplotlib
LABEL_COLORS_RGB = {
    "zero": (0.63, 0.63, 0.63),
    "one":  (0.00, 0.78, 0.00),
    "two":  (1.00, 0.39, 0.00),
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def contour_to_feature_vector(contour) -> np.ndarray:
    """Return a 1-D float32 feature vector for a single contour."""
    feats = contour_features(contour)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# DATA CHECK
# ---------------------------------------------------------------------------

def check_for_labeled_data(csv_path: str) -> bool:
    """
    Returns True if the labeled CSV exists and is non-empty.
    Prints a clear status message either way.
    """
    if csv_path and os.path.isfile(csv_path):
        try:
            with open(csv_path) as f:
                lines = [l for l in f if l.strip()]
            # Must have header + at least 1 data row
            if len(lines) > 1:
                print(f"[DATA CHECK] Labeled data FOUND: {csv_path}")
                print(f"             {len(lines)-1} labeled samples available.")
                return True
        except Exception as e:
            print(f"[DATA CHECK] Could not read {csv_path}: {e}")

    print("[DATA CHECK] Labeled data NOT FOUND.")
    print("             Expected: input/images/fly_count_labels.csv")
    print("             Columns : area, perimeter, aspect_ratio, extent, solidity, label")
    print("             Labels  : zero | one | two")
    print()
    print("             >> Falling back to RULE-BASED BASELINE <<")
    print("             See rule_based_predict() for details and limitations.")
    return False


# ---------------------------------------------------------------------------
# RULE-BASED BASELINE (used when no labeled data exists)
# ---------------------------------------------------------------------------

def rule_based_predict(contour) -> str:
    """
    Classify a contour using hard-coded area and solidity thresholds.

    Rules
    -----
    area < 80                    → "zero"  (noise blob)
    area >= 80 AND solidity > 0.85 → "one"   (compact single fly)
    area >= 80 AND solidity <= 0.85 → "two"  (merged pair, less compact)

    Why area?
      A single fruit fly occupies roughly 200-350 px² in a 320×240 arena.
      Two touching flies form a blob roughly twice as large (400-650 px²).
      Noise/dust blobs are typically smaller than 80 px².

    Why solidity (area / convex_hull_area)?
      A single fly body is a compact ellipse → solidity ~0.90–0.98.
      Two merged flies form a "figure-8" or irregular shape → solidity ~0.65–0.80.

    LIMITATIONS — this baseline will fail when:
      1. Camera distance or arena size differs from the training conditions,
         making the pixel area thresholds incorrect.
      2. Flies are partially occluded by the arena wall, reducing apparent area.
      3. Debris or reflections create blobs in the 80–350 px² range.
      4. Two flies are touching but happen to be well-aligned, giving high
         solidity and being misclassified as "one".
      5. A single fly is at the edge of the frame and clipped, reducing area.
      These cases require a trained Decision Tree with real labeled examples
      to distinguish reliably.
    """
    feats = contour_features(contour)
    area     = feats["area"]
    solidity = feats["solidity"]

    if area < 80:
        return "zero"
    elif solidity > 0.85:
        return "one"
    else:
        return "two"


# ---------------------------------------------------------------------------
# LOAD REAL LABELED DATA
# ---------------------------------------------------------------------------

def load_labeled_data(csv_path: str):
    """
    Load CSV with columns: area, perimeter, aspect_ratio, extent, solidity, label

    Returns
    -------
    X  : np.ndarray  (N, 5)
    y  : np.ndarray  (N,)   integer-encoded
    le : LabelEncoder
    """
    import csv
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    X = np.array([[float(r[name]) for name in FEATURE_NAMES]
                  for r in rows], dtype=np.float32)
    raw_labels = [r["label"].strip() for r in rows]

    le = LabelEncoder()
    le.fit(CLASS_NAMES)           # fix alphabetical order: one=0, two=1, zero=2
    y  = le.transform(raw_labels)
    return X, y, le


# ---------------------------------------------------------------------------
# SYNTHETIC DATASET (for Decision Tree demo when no real labels exist)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n_per_class: int = 150, random_state: int = 42):
    """
    Synthetic feature distributions calibrated to the synthetic video
    (320×240 arena, ellipse flies ~12×7 px → area ≈265 px²).

    NOTE: This is only used to demonstrate the Decision Tree pipeline.
    Performance on real video data will differ.
    """
    rng = np.random.default_rng(random_state)

    def sample(n, area_mu, area_sig, perim_mu, perim_sig,
               ar_mu, ar_sig, ext_mu, ext_sig, sol_mu, sol_sig):
        return np.column_stack([
            rng.normal(area_mu,  area_sig,  n).clip(10, 6000),
            rng.normal(perim_mu, perim_sig, n).clip(10, 500),
            rng.normal(ar_mu,    ar_sig,    n).clip(0.2, 4.0),
            rng.normal(ext_mu,   ext_sig,   n).clip(0.05, 1.0),
            rng.normal(sol_mu,   sol_sig,   n).clip(0.3, 1.0),
        ]).astype(np.float32)

    X_zero = sample(n_per_class,
                    40,  15,  30,  10,  1.10, 0.40, 0.55, 0.12, 0.70, 0.10)
    X_one  = sample(n_per_class,
                    265, 40,  75,  12,  1.67, 0.30, 0.62, 0.08, 0.97, 0.02)
    X_two  = sample(n_per_class,
                    530, 80, 140,  22,  2.50, 0.50, 0.50, 0.10, 0.72, 0.08)

    X = np.vstack([X_zero, X_one, X_two])

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    labels_str = (["zero"] * n_per_class +
                  ["one"]  * n_per_class +
                  ["two"]  * n_per_class)
    y = le.transform(labels_str)

    idx = rng.permutation(len(y))
    return X[idx], y[idx], le


# ---------------------------------------------------------------------------
# DECISION TREE TRAINING
# ---------------------------------------------------------------------------

def train_decision_tree(X: np.ndarray,
                        y: np.ndarray,
                        le: LabelEncoder,
                        max_depth: int = 5,
                        random_state: int = 42):
    """
    Train a DecisionTreeClassifier with a 75/25 train/test split.

    Hyperparameters
    ---------------
    max_depth=5      : limits tree depth to prevent overfitting on small datasets
    min_samples_leaf=5 : each leaf must contain at least 5 samples
    class_weight="balanced" : compensates for any class imbalance

    Returns
    -------
    clf, X_test, y_test, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------------------------

def predict_fly_count(clf, le, contour) -> str:
    """Predict label for a single contour using the trained Decision Tree."""
    feat_vec = contour_to_feature_vector(contour).reshape(1, -1)
    return le.inverse_transform([clf.predict(feat_vec)[0]])[0]


def label_to_count(label: str) -> int:
    """Convert a label string to an integer fly count."""
    return {"zero": 0, "one": 1, "two": 2}.get(label, 0)


# ---------------------------------------------------------------------------
# VISUALIZATION — contours + predicted labels on a frame
# ---------------------------------------------------------------------------

def visualize_predictions(frame: np.ndarray,
                           mask: np.ndarray,
                           contours: list,
                           labels: list,
                           total_count: int,
                           mode: str = "Decision Tree",
                           save_path: str = None):
    """
    Draw a 3-panel figure:
      Panel 1 — Grayscale frame
      Panel 2 — Binary mask (Otsu threshold)
      Panel 3 — Frame with contours drawn and labelled:
                   green outline  = "one" fly
                   orange outline = "two" flies merged
                   grey outline   = "zero" (noise)
                 Each contour shows:  label + (area)
                 Title shows total predicted fly count.

    Parameters
    ----------
    frame       : grayscale uint8 frame
    mask        : binary uint8 mask
    contours    : list of OpenCV contours
    labels      : list of str, one per contour
    total_count : sum of numeric counts across all contours
    mode        : "Decision Tree" or "Rule-based Baseline"
    save_path   : if given, save the figure here
    """
    # Build annotated colour image
    annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    for contour, label in zip(contours, labels):
        color_bgr = LABEL_COLORS_BGR.get(label, (255, 255, 255))
        thickness = 2 if label != "zero" else 1

        # Draw contour outline
        cv2.drawContours(annotated, [contour], -1, color_bgr, thickness)

        # Compute centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

        area = cv2.contourArea(contour)

        # Draw label text above centroid
        text = f"{label} ({int(area)})"
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)

        # Small background rectangle for readability
        cv2.rectangle(annotated,
                      (cx - 2, cy - th - 14),
                      (cx + tw + 2, cy - 10),
                      (0, 0, 0), -1)
        cv2.putText(annotated, text,
                    (cx, cy - 12),
                    font, font_scale, color_bgr, font_thick,
                    cv2.LINE_AA)

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Grayscale Frame", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Otsu Threshold Mask", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"Predicted fly count: {total_count}  "
        f"({len(contours)} contour(s))\n[{mode}]",
        fontsize=9
    )
    axes[2].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=LABEL_COLORS_RGB["one"],  label='one fly'),
        Patch(facecolor=LABEL_COLORS_RGB["two"],  label='two flies (merged)'),
        Patch(facecolor=LABEL_COLORS_RGB["zero"], label='noise (zero)'),
    ]
    axes[2].legend(handles=legend_elements,
                   loc="lower right", fontsize=7,
                   framealpha=0.7)

    fig.suptitle("FlyCount — Contour Prediction Visualization", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved visualization → {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(clf, X_test, y_test, y_pred, le, save_dir: str = "output"):
    """
    Print accuracy + classification report.
    Save confusion matrix PNG and decision tree PNG.
    """
    os.makedirs(save_dir, exist_ok=True)
    class_labels = le.classes_

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_labels).plot(ax=ax, colorbar=False)
    ax.set_title("Fly Count — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_fly_count.png")
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix  → {cm_path}")

    # Decision tree diagram
    fig2, ax2 = plt.subplots(figsize=(16, 7))
    plot_tree(clf,
              feature_names=FEATURE_NAMES,
              class_names=list(class_labels),
              filled=True, rounded=True,
              ax=ax2, fontsize=8)
    ax2.set_title("Decision Tree — Fly Count Classifier")
    tree_path = os.path.join(save_dir, "tree_fly_count.png")
    plt.savefig(tree_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved decision tree     → {tree_path}")

    # Text tree
    txt = export_text(clf, feature_names=FEATURE_NAMES)
    txt_path = os.path.join(save_dir, "tree_fly_count.txt")
    with open(txt_path, "w") as f:
        f.write(txt)
    print(f"  Saved tree text         → {txt_path}")

    return acc


# ---------------------------------------------------------------------------
# SAVE / LOAD MODEL
# ---------------------------------------------------------------------------

def save_model(clf, le, path: str = "models/fly_count_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, path)
    print(f"  Model saved → {path}")


def load_model(path: str = "models/fly_count_model.joblib"):
    bundle = joblib.load(path)
    return bundle["clf"], bundle["le"]


# ---------------------------------------------------------------------------
# TEST ON A SAMPLE FRAME
# ---------------------------------------------------------------------------

def test_on_sample_frame(video_path: str,
                         clf=None,
                         le=None,
                         use_rule_based: bool = False,
                         save_dir: str = "output"):
    """
    Run the full pipeline on one frame from the video:
      grayscale → threshold → contours → predict → visualize

    If clf is None (no trained model), uses the rule-based baseline.
    """
    cap, props = load_video(video_path)
    background = compute_background(cap, n_samples=min(50, props["frame_count"]))

    # Take the 5th frame (index 4) — early enough to be stable, not frame 0
    frame = None
    for i, f in enumerate(read_frames(cap, max_frames=10)):
        frame = f
        if i == 4:
            break
    cap.release()

    if frame is None:
        print("  [WARN] Could not read a sample frame.")
        return

    mask     = threshold_frame(frame, background)
    contours = extract_contours(mask)

    labels = []
    for c in contours:
        if use_rule_based or (clf is None):
            label = rule_based_predict(c)
        else:
            label = predict_fly_count(clf, le, c)
        labels.append(label)

    total_count = sum(label_to_count(l) for l in labels)
    mode = "Rule-based Baseline" if (use_rule_based or clf is None) else "Decision Tree"

    print(f"\n[SAMPLE FRAME TEST]  mode={mode}")
    print(f"  Contours found : {len(contours)}")
    for i, (c, lbl) in enumerate(zip(contours, labels)):
        feats = contour_features(c)
        print(f"  Contour {i}: label={lbl:4s}  "
              f"area={feats['area']:6.1f}  "
              f"solidity={feats['solidity']:.2f}  "
              f"AR={feats['aspect_ratio']:.2f}")
    print(f"  Total predicted fly count: {total_count}")

    save_path = os.path.join(save_dir, "flycount_sample_frame.png")
    visualize_predictions(frame, mask, contours, labels,
                          total_count, mode=mode,
                          save_path=save_path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(data_csv: str = None,
         video_path: str = None,
         save_dir: str = "output"):

    print("=" * 58)
    print("  FlyCount Module — Stage 1")
    print("=" * 58)

    # ── Step 1: check for labeled data ──────────────────────────────────────
    has_real_data = check_for_labeled_data(data_csv)

    clf, le = None, None

    if has_real_data:
        # ── Step 2a: train Decision Tree on real labels ──────────────────
        print("\n[TRAIN] Loading labeled data ...")
        X, y, le = load_labeled_data(data_csv)
        dist = {le.classes_[i]: int((y == i).sum()) for i in range(len(le.classes_))}
        print(f"        {len(y)} samples  |  distribution: {dist}")

        print("\n[TRAIN] Fitting Decision Tree (max_depth=5) ...")
        clf, X_test, y_test, y_pred = train_decision_tree(X, y, le)

        print("\n[EVAL]  Evaluating on held-out test set (25%) ...")
        acc = evaluate(clf, X_test, y_test, y_pred, le, save_dir)

        print("\n[SAVE]  Saving model ...")
        save_model(clf, le)

        print(f"\n[DONE]  Decision Tree trained.  Test accuracy: {acc*100:.2f}%")

    else:
        # ── Step 2b: rule-based baseline + Decision Tree demo ────────────
        print("\n[MODE]  RULE-BASED BASELINE active (no labeled data).")
        print("        Rules: area<80→zero | area≥80 & solidity>0.85→one | else→two")
        print()
        print("        LIMITATIONS of the rule-based baseline:")
        print("          1. Area thresholds break if camera distance changes.")
        print("          2. Occluded or clipped flies are misclassified.")
        print("          3. Debris in the 80-350 px² range causes false positives.")
        print("          4. Well-aligned merged flies fool the solidity rule.")
        print("          5. No generalisation — any arena/lighting change breaks it.")
        print()
        print("[DEMO]  Also running Decision Tree on SYNTHETIC data for comparison.")

        X, y, le = make_synthetic_dataset(n_per_class=150)
        print(f"        Generated {len(y)} synthetic samples.")

        clf, X_test, y_test, y_pred = train_decision_tree(X, y, le)

        print("\n[EVAL]  Decision Tree on synthetic data:")
        acc = evaluate(clf, X_test, y_test, y_pred, le, save_dir)
        save_model(clf, le)

        print(f"\n        Synthetic accuracy: {acc*100:.2f}%")
        print("        NOTE: this accuracy is on synthetic data only.")
        print("              Provide fly_count_labels.csv for real training.")

    # ── Step 3: test on a sample frame ──────────────────────────────────────
    print("\n[TEST]  Running on a sample video frame ...")

    if video_path is None or not os.path.exists(video_path):
        print("        No video supplied — generating synthetic demo video ...")
        video_path = make_synthetic_video()

    # Show both modes side by side when no real data exists
    if not has_real_data:
        print("        >> Rule-based baseline prediction:")
        test_on_sample_frame(video_path,
                             clf=None, le=None,
                             use_rule_based=True,
                             save_dir=save_dir)
        # Override save path name so both images are kept
        import shutil
        rb_src = os.path.join(save_dir, "flycount_sample_frame.png")
        rb_dst = os.path.join(save_dir, "flycount_sample_frame_rulebased.png")
        if os.path.exists(rb_src):
            shutil.copy(rb_src, rb_dst)

        print("\n        >> Decision Tree (synthetic) prediction:")
        test_on_sample_frame(video_path,
                             clf=clf, le=le,
                             use_rule_based=False,
                             save_dir=save_dir)
    else:
        test_on_sample_frame(video_path,
                             clf=clf, le=le,
                             use_rule_based=False,
                             save_dir=save_dir)

    print("\n[DONE]  FlyCount module complete.")
    print(f"        Outputs saved to: {save_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlyCount: contour-based fly counting with Decision Tree")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to fly_count_labels.csv  "
                             "(omit to use rule-based baseline + synthetic DT demo)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input .mp4 for sample-frame test")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save all outputs  (default: output/)")
    args = parser.parse_args()

    main(data_csv=args.data,
         video_path=args.video,
         save_dir=args.save_dir)
