"""
orientation.py
--------------
Body orientation estimation for fruit fly video analysis.

The orientation problem has two stages:

  Stage 1 -- Body-axis angle (image moments, always available)
  -----------------------------------------------------------
  cv2.moments() on the binary fly mask gives the second-order central
  moments mu20, mu02, mu11.  The principal axis of the ellipse that has
  the same second moments as the blob is:

      theta = 0.5 * arctan2(2*mu11, mu20 - mu02)

  This yields an angle in (-pi/2, pi/2) -- i.e. it only tells you the
  *axis*, not the *direction* along that axis.  Both theta and theta+pi
  are equally valid solutions from the moment equation, so the result
  has a 180-degree ambiguity.

  Stage 2 -- Ambiguity resolution (HOG + PCA + Logistic Regression)
  -----------------------------------------------------------------
  To resolve which end is the head and which is the abdomen, we use
  gradient appearance.  HOG features capture the local edge structure
  of the fly patch; the head region has a different texture signature
  than the abdomen.  A Logistic Regression classifier (trained on
  labeled patches) learns to predict whether the current moment angle
  needs to be flipped by pi.

  Label convention for the CSV:
    flip = 0  ->  moment angle is already correct (head points in the
                 direction the angle arrow points)
    flip = 1  ->  moment angle needs pi added (head points the other way)

  Data policy
  -----------
  This module checks for orientation labels at startup.
  If the CSV is absent: Stage 1 (moments baseline) runs fully.
                        Stage 2 (disambiguation) is skipped cleanly.
                        No labels are fabricated.
  If the CSV is present: full two-stage pipeline trains and evaluates.

Expected CSV  (input/images/orientation_labels.csv)
  patch_path, moment_angle_rad, flip
  Each row is one labeled fly patch.

Usage
-----
    python orientation.py                                  # baseline demo
    python orientation.py --data input/images/orientation_labels.csv
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from skimage.feature import hog as skimage_hog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_H = 64
PATCH_W = 64

HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

MODEL_PATH   = "models/orientation_model.joblib"
DEFAULT_CSV  = "input/images/orientation_labels.csv"


# ---------------------------------------------------------------------------
# DATA CHECK
# ---------------------------------------------------------------------------

def check_for_labeled_data(csv_path: str) -> bool:
    """
    Return True only if the CSV exists, is non-empty, and has both
    flip=0 and flip=1 examples.  Prints a clear status in all cases.
    """
    print("[DATA CHECK] Looking for orientation labels ...")
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
        print("[DATA CHECK] File exists but contains no data rows.")
        _print_data_instructions(csv_path)
        return False

    required_cols = {"patch_path", "moment_angle_rad", "flip"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        print(f"[DATA CHECK] Missing columns: {missing}")
        _print_data_instructions(csv_path)
        return False

    flips = {int(r["flip"]) for r in rows}
    if not {0, 1}.issubset(flips):
        print(f"[DATA CHECK] Need both flip=0 and flip=1 examples. "
              f"Found: {flips}")
        _print_data_instructions(csv_path)
        return False

    n0 = sum(1 for r in rows if int(r["flip"]) == 0)
    n1 = sum(1 for r in rows if int(r["flip"]) == 1)
    print(f"[DATA CHECK] FOUND -- {len(rows)} rows  "
          f"(no-flip: {n0}, flip: {n1})")
    return True


def _print_data_instructions(csv_path: str):
    print()
    print("  Stage 2 (ambiguity resolution) requires labeled patches.")
    print("  No synthetic labels will be fabricated.")
    print(f"  Expected file: {csv_path}")
    print()
    print("  CSV columns:")
    print("    patch_path        -- path to a 64×64 grayscale PNG of the fly")
    print("    moment_angle_rad  -- body-axis angle from image moments (radians)")
    print("    flip              -- 0 if angle is correct, 1 if pi must be added")
    print()
    print("  How to collect labels:")
    print("    1. Download the CS229 labeled dataset (see README.md).")
    print("    2. For each labeled frame, run moments_orientation() to get the")
    print("       raw moment angle.")
    print("    3. Manually verify: if the arrow points toward the abdomen")
    print("       instead of the head, set flip=1.")
    print()
    print("  Only the moments baseline (Stage 1) will run without this data.")


# ---------------------------------------------------------------------------
# STAGE 1 -- IMAGE MOMENTS BASELINE
# ---------------------------------------------------------------------------

def moments_orientation(patch: np.ndarray) -> float:
    """
    Compute the body-axis angle from second-order image moments.

    The fly body is darker than the background.  We invert the patch
    so the fly pixels are bright before computing moments -- this ensures
    mu20 and mu02 describe the fly blob, not the background.

    Mathematics
    -----------
    Given central moments mu20, mu02, mu11, the principal axis angle is:

        theta = 0.5 * arctan2(2 * mu11, mu20 - mu02)

    Result is in (-pi/2, pi/2).  The 180-degree ambiguity is NOT resolved
    here; both theta and theta+pi are geometrically equivalent from moments
    alone.

    Parameters
    ----------
    patch : grayscale uint8 image  (fly region, any size)

    Returns
    -------
    angle : float -- body axis angle in radians, range (-pi/2, pi/2)
            Returns 0.0 if moments are degenerate (circular blob).
    """
    # Invert: fly is dark on light background -> make fly bright
    _, binary = cv2.threshold(patch, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    M = cv2.moments(binary)

    denom = M["mu20"] - M["mu02"]
    numer = 2.0 * M["mu11"]

    # Degenerate case: circular blob -- no dominant axis
    if abs(denom) < 1e-6 and abs(numer) < 1e-6:
        return 0.0

    angle = 0.5 * np.arctan2(numer, denom)
    return float(angle)


# ---------------------------------------------------------------------------
# STAGE 2 -- HOG FEATURE EXTRACTION FOR DISAMBIGUATION
# ---------------------------------------------------------------------------

def extract_hog_features(patch: np.ndarray) -> np.ndarray:
    """
    Compute HOG features from a fly patch for ambiguity resolution.

    The patch is resized to PATCH_H × PATCH_W before HOG computation so
    all feature vectors have the same length regardless of input size.

    HOG encodes local gradient orientations in overlapping blocks.  The
    head and abdomen regions of Drosophila have different gradient patterns
    (head: rounder, darker; abdomen: tapered, lighter stripes), which
    allows a classifier to learn which end is which.

    Returns
    -------
    feat_vec : 1-D float64 array
    """
    resized = cv2.resize(patch, (PATCH_W, PATCH_H))
    feat_vec = skimage_hog(
        resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat_vec


# ---------------------------------------------------------------------------
# LOAD LABELED DATA
# ---------------------------------------------------------------------------

def load_labeled_data(csv_path: str):
    """
    Load orientation labels and patch images.

    Returns
    -------
    patches       : list of np.ndarray  (H, W) uint8
    moment_angles : np.ndarray  (N,)   raw moment angles (radians)
    flips         : np.ndarray  (N,)   int  0 or 1
    """
    import csv

    patches, angles, flips = [], [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = cv2.imread(row["patch_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Cannot load patch: {row['patch_path']}")
                continue
            patches.append(cv2.resize(img, (PATCH_W, PATCH_H)))
            angles.append(float(row["moment_angle_rad"]))
            flips.append(int(row["flip"]))

    return patches, np.array(angles, dtype=np.float32), np.array(flips, dtype=np.int32)


# ---------------------------------------------------------------------------
# TRAIN -- PCA + Logistic Regression disambiguation
# ---------------------------------------------------------------------------

def train_disambiguation_model(patches: list,
                                flips: np.ndarray,
                                random_state: int = 42):
    """
    Train a PCA + LogisticRegression pipeline to predict flip (0 or 1).

    Why PCA?
      HOG produces a high-dimensional vector (~3000+ dims for 64×64).
      PCA reduces this to the top 30 principal components -- captures the
      dominant variance while preventing overfitting on small datasets.

    Why Logistic Regression?
      Binary classification (flip vs no-flip).  L2 regularization guards
      against overfitting.  Produces calibrated probabilities.

    Returns
    -------
    model   : fitted Pipeline(PCA, LogisticRegression)
    X_test  : test HOG features
    y_test  : true flip labels
    y_pred  : predicted flip labels
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    X = np.array([extract_hog_features(p) for p in patches], dtype=np.float64)
    y = flips

    n_components = min(30, X.shape[1], X.shape[0] - 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# EVALUATE DISAMBIGUATION
# ---------------------------------------------------------------------------

def evaluate_disambiguation(y_test, y_pred, save_dir: str = "output"):
    """Print accuracy and save a confusion matrix."""
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    os.makedirs(save_dir, exist_ok=True)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Disambiguation accuracy : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["no-flip (0)", "flip (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["no-flip", "flip"]).plot(
        ax=ax, colorbar=False)
    ax.set_title("Orientation Disambiguation -- Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix_orientation.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {path}")
    return acc


# ---------------------------------------------------------------------------
# SAVE / LOAD MODEL
# ---------------------------------------------------------------------------

def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved -> {path}")


def load_model(path: str = MODEL_PATH):
    if not os.path.isfile(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# PREDICT ORIENTATION  (used by main.py)
# ---------------------------------------------------------------------------

def predict_orientation(patch: np.ndarray, model=None) -> float:
    """
    Estimate body orientation angle for a single fly patch.

    If model is None (no disambiguation model trained): returns the raw
    moment angle, which has a 180-degree ambiguity.

    If model is provided: the HOG-based classifier predicts whether the
    moment angle needs to be flipped by pi, giving a fully directed angle.

    Parameters
    ----------
    patch : grayscale uint8 image of the fly region
    model : fitted Pipeline (PCA + LogReg) or None

    Returns
    -------
    angle : float -- estimated body orientation in radians
                    range (-pi/2, pi/2) without model
                    range (-pi, pi)     with model
    """
    angle = moments_orientation(patch)

    if model is not None:
        feat = extract_hog_features(patch).reshape(1, -1)
        flip = int(model.predict(feat)[0])
        if flip == 1:
            angle = angle + np.pi
            # Normalize to (-pi, pi]
            if angle > np.pi:
                angle -= 2 * np.pi

    return float(angle)


# ---------------------------------------------------------------------------
# VISUALIZATION -- body axis drawn on the patch and on the full frame
# ---------------------------------------------------------------------------

def visualize_orientation(patch: np.ndarray,
                          angle: float,
                          has_disambiguation: bool = False,
                          save_path: str = None):
    """
    Draw the estimated body axis as an arrow on the fly patch.

    The arrow points in the direction of the estimated head end.
    Without a disambiguation model the arrow could be 180 deg wrong --
    this is clearly indicated in the plot title.

    Parameters
    ----------
    patch                : grayscale uint8 fly patch
    angle                : estimated orientation angle (radians)
    has_disambiguation   : True if the 180 deg ambiguity has been resolved
    save_path            : file path to save the figure (optional)
    """
    h, w = patch.shape[:2]
    cx, cy = w // 2, h // 2
    arrow_len = min(h, w) * 0.38

    # Tip of the arrow in image coordinates (y-axis flipped)
    tip_x = int(cx + arrow_len * np.cos(angle))
    tip_y = int(cy - arrow_len * np.sin(angle))
    tip_x = np.clip(tip_x, 0, w - 1)
    tip_y = np.clip(tip_y, 0, h - 1)

    # Also draw the opposite end (tail) to show full axis
    tail_x = int(cx - arrow_len * np.cos(angle))
    tail_y = int(cy + arrow_len * np.sin(angle))
    tail_x = np.clip(tail_x, 0, w - 1)
    tail_y = np.clip(tail_y, 0, h - 1)

    vis = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # Draw axis line in yellow
    cv2.line(vis, (tail_x, tail_y), (tip_x, tip_y), (0, 220, 220), 1)

    # Arrow tip in green (head direction) if disambiguated, else red (ambiguous)
    arrow_color = (0, 200, 0) if has_disambiguation else (0, 60, 220)
    cv2.arrowedLine(vis, (cx, cy), (tip_x, tip_y),
                    arrow_color, 2, tipLength=0.35)

    # Draw centre dot
    cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    axes[0].imshow(patch, cmap="gray")
    axes[0].set_title("Fly Patch (grayscale)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    if has_disambiguation:
        title = (f"Body axis -- angle {np.degrees(angle):.1f} deg\n"
                 f"(180-deg ambiguity RESOLVED by HOG + Logistic Regression)")
        axes[1].set_title(title, fontsize=8, color="darkgreen")
    else:
        title = (f"Body axis -- angle {np.degrees(angle):.1f} deg\n"
                 f"[!] 180-deg ambiguity NOT resolved (no labeled data)\n"
                 f"Arrow may point toward head OR abdomen")
        axes[1].set_title(title, fontsize=8, color="firebrick")

    axes[1].axis("off")

    fig.suptitle("Orientation Estimation", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved orientation visualization -> {save_path}")

    plt.close(fig)


def visualize_orientation_on_frame(frame: np.ndarray,
                                   contours: list,
                                   angles: list,
                                   has_disambiguation: bool = False,
                                   save_path: str = None):
    """
    Draw body-axis arrows for all detected flies on the full frame.

    Parameters
    ----------
    frame              : grayscale uint8 full video frame
    contours           : list of OpenCV contours (one per fly)
    angles             : list of float (estimated orientation per contour)
    has_disambiguation : whether the 180 deg flip has been resolved
    save_path          : optional save path
    """
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    arrow_color = (0, 200, 0) if has_disambiguation else (0, 60, 220)

    for contour, angle in zip(contours, angles):
        M = cv2.moments(contour)
        if M["m00"] < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Scale arrow to contour size
        x, y, bw, bh = cv2.boundingRect(contour)
        arrow_len = max(bw, bh) * 0.6

        tip_x = int(cx + arrow_len * np.cos(angle))
        tip_y = int(cy - arrow_len * np.sin(angle))
        tip_x = np.clip(tip_x, 0, frame.shape[1] - 1)
        tip_y = np.clip(tip_y, 0, frame.shape[0] - 1)

        cv2.drawContours(vis, [contour], -1, (180, 180, 180), 1)
        cv2.arrowedLine(vis, (cx, cy), (tip_x, tip_y),
                        arrow_color, 2, tipLength=0.3)
        cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Original Frame", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    amb_note = ("resolved" if has_disambiguation
                else "[!] 180-deg ambiguity unresolved")
    axes[1].set_title(f"Body-Axis Arrows ({amb_note})", fontsize=9)
    axes[1].axis("off")

    fig.suptitle("Orientation Estimation -- Frame Visualization", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved frame visualization -> {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# BASELINE DEMO  (run when no labeled data is found)
# ---------------------------------------------------------------------------

def _run_baseline_demo(save_dir: str):
    """
    Create a synthetic fly patch, run the moments baseline, and save the
    visualization.  Shows exactly what Stage 1 produces and why Stage 2
    is needed.
    """
    print()
    print("[BASELINE DEMO]  Creating synthetic fly patch ...")

    # Synthetic patch: dark elongated ellipse on light background
    patch = np.full((PATCH_H, PATCH_W), 210, dtype=np.uint8)
    true_angle_deg = 35.0
    true_angle_rad = np.radians(true_angle_deg)
    cx, cy = PATCH_W // 2, PATCH_H // 2

    # Draw body ellipse rotated by true_angle
    cv2.ellipse(patch, (cx, cy), (22, 8), -true_angle_deg,
                0, 360, 35, -1)

    # Add a small "head" bump to give asymmetry (moments can't see this)
    hx = int(cx + 18 * np.cos(true_angle_rad))
    hy = int(cy - 18 * np.sin(true_angle_rad))
    hx = np.clip(hx, 4, PATCH_W - 4)
    hy = np.clip(hy, 4, PATCH_H - 4)
    cv2.circle(patch, (hx, hy), 4, 35, -1)

    rng = np.random.default_rng(7)
    noise = rng.integers(-8, 8, patch.shape, dtype=np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    angle = moments_orientation(patch)
    print(f"  True body axis : {true_angle_deg:.1f} deg")
    print(f"  Moments result : {np.degrees(angle):.1f} deg  "
          f"(or {np.degrees(angle) + 180:.1f} deg -- ambiguous)")

    vis_path = os.path.join(save_dir, "orientation_baseline_demo.png")
    visualize_orientation(patch, angle,
                          has_disambiguation=False,
                          save_path=vis_path)

    print()
    print("  LIMITATIONS of the moments-only baseline:")
    print("  1. 180-deg ambiguity: moments give the axis, not the direction.")
    print("     Both theta and theta+pi are indistinguishable from shape alone.")
    print("  2. Pose sensitivity: a fly curled or at the arena edge gives")
    print("     a distorted ellipse, rotating the apparent axis by 10-30 deg.")
    print("  3. Merged blobs: when two flies touch, the combined moment axis")
    print("     points between the two bodies, not along either fly.")
    print("  4. No temporal continuity: each frame is treated independently;")
    print("     small per-frame noise can cause 180-deg flips between frames.")
    print("  5. Resolution to Stage 2 requires labeled patches with flip=0/1.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(csv_path: str = DEFAULT_CSV, save_dir: str = "output"):
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 58)
    print("  Orientation Estimation Module")
    print("=" * 58)
    print()

    # ── Step 1: data check ───────────────────────────────────────────────────
    has_data = check_for_labeled_data(csv_path)

    if not has_data:
        # Stage 1 only -- moments baseline with visualization
        _run_baseline_demo(save_dir)
        print()
        print("[STATUS] Stage 1 (moments baseline) is operational.")
        print("         Stage 2 (HOG + PCA + LogReg disambiguation) requires")
        print(f"         labeled data at: {csv_path}")
        print("         predict_orientation() will use Stage 1 only until then.")
        return

    # ── Step 2: load data ────────────────────────────────────────────────────
    print()
    print("[LOAD]  Reading labeled patches ...")
    patches, moment_angles, flips = load_labeled_data(csv_path)
    print(f"        {len(patches)} patches loaded  "
          f"(no-flip: {(flips==0).sum()}, flip: {(flips==1).sum()})")

    # ── Step 3: train disambiguation model ──────────────────────────────────
    print()
    print("[TRAIN] Extracting HOG features and fitting PCA + LogisticRegression ...")
    model, X_test, y_test, y_pred = train_disambiguation_model(patches, flips)
    expl = model.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"        PCA: top 30 components retain {expl*100:.1f}% of variance.")

    # ── Step 4: evaluate ─────────────────────────────────────────────────────
    print()
    print("[EVAL]  Evaluating on held-out test set (25%) ...")
    acc = evaluate_disambiguation(y_test, y_pred, save_dir)

    # ── Step 5: visualize a sample patch ────────────────────────────────────
    sample_patch  = patches[0]
    raw_angle     = moment_angles[0]
    resolved      = predict_orientation(sample_patch, model=model)
    vis_path      = os.path.join(save_dir, "orientation_sample.png")
    visualize_orientation(sample_patch, resolved,
                          has_disambiguation=True,
                          save_path=vis_path)

    # ── Step 6: save ─────────────────────────────────────────────────────────
    print()
    print("[SAVE]  Saving model ...")
    save_model(model)

    print()
    print(f"[DONE]  Orientation module trained.  "
          f"Disambiguation accuracy: {acc*100:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orientation estimation: moments baseline + HOG disambiguation")
    parser.add_argument("--data", type=str, default=DEFAULT_CSV,
                        help=f"Path to labeled CSV  (default: {DEFAULT_CSV})")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save outputs  (default: output/)")
    args = parser.parse_args()
    main(csv_path=args.data, save_dir=args.save_dir)
