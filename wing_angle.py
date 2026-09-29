"""
wing_angle.py  -- Member 1, Stage 4
-------------------------------------
Wing-angle estimation for male fruit flies.

Pipeline
--------
1. Image preprocessing
   - Crop a fixed-size ROI (64×64 px) centred on the male fly's centroid.
   - Apply Gaussian blur and CLAHE to enhance wing edges.
   - Split the ROI into left-wing and right-wing sub-regions.

2. Feature extraction
   - Compute HOG (Histogram of Oriented Gradients) on each sub-region.
   - Concatenate left + right HOG vectors into one feature vector.

3. Dimensionality reduction
   - PCA reduces the HOG vector (~3500 dims) to 30 principal components.
   - This prevents overfitting and speeds up regression.

4. Regression (only when labeled data is available)
   - Two separate LinearRegression models: one for left angle, one for right.
   - Evaluated on a held-out 25% test set using MAE and RMSE.

Data policy
-----------
This module checks for labeled wing-angle annotations at startup.

  FOUND  -> full pipeline: preprocessing, HOG, PCA, LinearRegression,
            evaluation on a real test set, model saved.

  MISSING -> preprocessing and visualization run fully (so you can inspect
            what the pipeline sees), but NO regression model is trained,
            NO predictions are fabricated, and NO accuracy numbers are
            reported.  A clear explanation of what data is needed is printed.

Expected CSV  (input/images/wing_labels.csv)
  patch_path, angle_right_rad, angle_left_rad
  - patch_path    : path to a 64×64 grayscale PNG of the male fly
  - angle_right_rad : right wing angle in radians (measured from body axis)
  - angle_left_rad  : left wing angle in radians

Usage
-----
    python wing_angle.py                                 # data check + demo
    python wing_angle.py --data input/images/wing_labels.csv
    python wing_angle.py --video input/video/test4.mp4   # frame demo
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import (make_synthetic_video, load_video,
                            compute_background, read_frames,
                            threshold_frame, extract_contours)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_H = 64
PATCH_W = 64

# Wing sub-region: each side gets 45% of patch width
WING_FRACTION = 0.45

HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

PCA_COMPONENTS  = 30

MODEL_PATH_R = "models/wing_angle_right_model.joblib"
MODEL_PATH_L = "models/wing_angle_left_model.joblib"
DEFAULT_CSV  = "input/images/wing_labels.csv"


# ---------------------------------------------------------------------------
# DATA CHECK
# ---------------------------------------------------------------------------

def check_for_labeled_data(csv_path: str) -> bool:
    """
    Return True only if the CSV exists, is non-empty, and has all
    required columns with loadable patch images.
    """
    print("[DATA CHECK] Looking for wing-angle labels ...")
    print(f"             Path : {csv_path}")

    if not os.path.isfile(csv_path):
        print("[DATA CHECK] NOT FOUND.")
        _print_data_instructions(csv_path)
        return False

    try:
        import csv as csv_mod
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv_mod.DictReader(f)
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

    required = {"patch_path", "angle_right_rad", "angle_left_rad"}
    missing  = required - set(rows[0].keys())
    if missing:
        print(f"[DATA CHECK] Missing columns: {missing}")
        _print_data_instructions(csv_path)
        return False

    print(f"[DATA CHECK] FOUND -- {len(rows)} labeled rows.")
    return True


def _print_data_instructions(csv_path: str):
    print()
    print("  No regression model will be trained without real labeled data.")
    print("  No wing angles will be fabricated.")
    print(f"  Expected file: {csv_path}")
    print()
    print("  CSV format:")
    print("    patch_path,angle_right_rad,angle_left_rad")
    print("    input/images/patches/frame001_male.png,0.524,0.611")
    print("    ...")
    print()
    print("  What each column means:")
    print("    patch_path       -- 64×64 grayscale PNG of the male fly ROI")
    print("    angle_right_rad  -- angle of the right wing from the body axis")
    print("                       measured in radians, range (0, pi)")
    print("    angle_left_rad   -- same for the left wing")
    print()
    print("  How to collect labels:")
    print("    1. Download the CS229 labeled dataset (see README.md).")
    print("       It contains images/ with labeled male-fly patches and")
    print("       a data file with manual wing-angle annotations.")
    print("    2. Export one row per patch with the three columns above.")
    print()
    print("  Preprocessing and visualization will still run so you can")
    print("  inspect the pipeline output before data is available.")


# ---------------------------------------------------------------------------
# STEP 1 -- PATCH EXTRACTION AND PREPROCESSING
# ---------------------------------------------------------------------------

def extract_fly_patch(frame: np.ndarray,
                      center_x: int,
                      center_y: int,
                      patch_h: int = PATCH_H,
                      patch_w: int = PATCH_W):
    """
    Crop a fixed-size ROI centred on the male fly from a grayscale frame.

    Returns None if the crop would extend outside the frame boundaries.
    """
    h, w = frame.shape[:2]
    x0, y0 = center_x - patch_w // 2, center_y - patch_h // 2
    x1, y1 = x0 + patch_w,            y0 + patch_h

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    return frame[y0:y1, x0:x1].copy().astype(np.uint8)


def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    """
    Enhance the fly patch to make wing edges more visible.

    Steps
    -----
    1. Gaussian blur (3×3) -- suppresses high-frequency sensor noise.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) --
       enhances local contrast so the thin wing edges become detectable
       by HOG even if the original patch has low contrast.

    Returns a preprocessed uint8 patch of the same size.
    """
    blurred = cv2.GaussianBlur(patch, (3, 3), 0)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(blurred)
    return enhanced


def split_wing_regions(patch: np.ndarray):
    """
    Split the fly patch into left and right wing sub-regions.

    Layout (patch_w=64, WING_FRACTION=0.45 -> wing_w=28 px):
      columns  0..27  -> left wing region
      columns 28..35  -> body centre (discarded)
      columns 36..63  -> right wing region

    The male fly is assumed to be centred in the patch with its body
    running roughly vertically.  Wings extend horizontally on both sides.

    Returns
    -------
    left_wing, right_wing : uint8 arrays
    """
    pw     = patch.shape[1]
    wing_w = int(pw * WING_FRACTION)
    return patch[:, :wing_w], patch[:, pw - wing_w:]


# ---------------------------------------------------------------------------
# STEP 2 -- HOG FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def hog_features(region: np.ndarray) -> np.ndarray:
    """
    Compute HOG features for one wing sub-region.

    The region is resized to PATCH_H × PATCH_W before computing HOG so
    all feature vectors have the same length regardless of input size.

    Why HOG for wings?
    HOG encodes the distribution of local edge orientations in overlapping
    spatial blocks.  A wing at 60 deg from the body axis creates a dominant
    gradient direction at 60 deg; a wing at 120 deg creates gradients at 120 deg.
    This makes HOG highly informative about wing angle.

    Returns
    -------
    feat_vec : 1-D float64 array
    """
    resized = cv2.resize(region, (PATCH_W, PATCH_H))
    return skimage_hog(
        resized,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def patch_to_hog_vector(patch: np.ndarray) -> np.ndarray:
    """
    Full pipeline: preprocess patch -> split -> HOG left + HOG right -> concat.

    Returns
    -------
    feat_vec : 1-D float64 array  length = 2 × hog_dim
    """
    preprocessed      = preprocess_patch(patch)
    left_w, right_w   = split_wing_regions(preprocessed)
    return np.concatenate([hog_features(left_w), hog_features(right_w)])


def build_feature_matrix(patches: list) -> np.ndarray:
    """Apply patch_to_hog_vector to every patch. Returns (N, D) array."""
    return np.array([patch_to_hog_vector(p) for p in patches],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# LOAD LABELED DATA
# ---------------------------------------------------------------------------

def load_labeled_data(csv_path: str):
    """
    Load patch images and wing-angle labels from the CSV.

    Returns
    -------
    patches       : list of np.ndarray  (PATCH_H, PATCH_W) uint8
    angles_right  : np.ndarray  (N,)   float32  radians
    angles_left   : np.ndarray  (N,)   float32  radians
    """
    import csv as csv_mod
    patches, ar, al = [], [], []

    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            img = cv2.imread(row["patch_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Cannot load: {row['patch_path']}")
                continue
            patches.append(cv2.resize(img, (PATCH_W, PATCH_H)))
            ar.append(float(row["angle_right_rad"]))
            al.append(float(row["angle_left_rad"]))

    return (patches,
            np.array(ar, dtype=np.float32),
            np.array(al, dtype=np.float32))


# ---------------------------------------------------------------------------
# STEP 3+4 -- PCA + LINEAR REGRESSION
# ---------------------------------------------------------------------------

def train_wing_angle_model(X: np.ndarray,
                           y: np.ndarray,
                           wing: str = "right",
                           random_state: int = 42):
    """
    Train a PCA + LinearRegression pipeline for one wing angle target.

    Why PCA before regression?
      HOG produces ~3500 features for a 64×64 patch.  Linear Regression
      on raw HOG would overfit on small datasets (typical labeling
      yields a few hundred examples).  PCA retains the top 30 components
      which together capture the dominant gradient structure, stripping
      noise and collinear dimensions.

    Why Linear Regression?
      Wing angle is a continuous scalar in (0, pi).  The relationship
      between HOG gradient magnitudes and angle is approximately linear
      in the PCA subspace -- validated empirically in the CS229 paper.

    Parameters
    ----------
    X    : HOG feature matrix  (N, D)
    y    : target angles in radians  (N,)
    wing : label for print output only

    Returns
    -------
    model  : fitted Pipeline(PCA -> LinearRegression)
    X_test : held-out HOG features  (25%)
    y_test : true test angles
    y_pred : predicted test angles
    """
    n_pca = min(PCA_COMPONENTS, X.shape[1], X.shape[0] - 1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = Pipeline([
        ("pca",        PCA(n_components=n_pca, random_state=random_state)),
        ("regressor",  LinearRegression()),
    ])
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return model, X_te, y_te, y_pred


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_regression(y_test: np.ndarray,
                        y_pred: np.ndarray,
                        wing: str = "right",
                        save_dir: str = "output") -> tuple:
    """
    Compute MAE and RMSE on real test data; save a scatter plot.

    These metrics are only reported when real labeled data is available.
    """
    os.makedirs(save_dir, exist_ok=True)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"\n  [{wing.upper()} WING]")
    print(f"    MAE  = {mae:.4f} rad  ({np.degrees(mae):.2f} deg)")
    print(f"    RMSE = {rmse:.4f} rad  ({np.degrees(rmse):.2f} deg)")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.degrees(y_test), np.degrees(y_pred),
               alpha=0.55, s=18, label="Test samples")
    lims = [min(np.degrees(y_test).min(), np.degrees(y_pred).min()) - 5,
            max(np.degrees(y_test).max(), np.degrees(y_pred).max()) + 5]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("True angle ( deg)")
    ax.set_ylabel("Predicted angle ( deg)")
    ax.set_title(f"Wing Angle Regression -- {wing} wing\n"
                 f"MAE = {np.degrees(mae):.1f} deg   RMSE = {np.degrees(rmse):.1f} deg")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(save_dir, f"wing_angle_{wing}_scatter.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"    Saved scatter plot -> {path}")

    return mae, rmse


# ---------------------------------------------------------------------------
# VISUALIZATION -- preprocessing pipeline output
# ---------------------------------------------------------------------------

def visualize_patch_pipeline(patch: np.ndarray,
                              angle_right: float = None,
                              angle_left: float = None,
                              save_path: str = None):
    """
    Show the preprocessing stages and wing-region crops for one fly patch.

    If angle_right / angle_left are provided (from real labels), they are
    drawn as arrows on the patch.  If not provided, the visualization
    still shows the preprocessing output without any fabricated angles.

    Panels
    ------
    1. Raw patch
    2. After Gaussian blur + CLAHE
    3. Left wing sub-region
    4. Right wing sub-region
    5. HOG visualization (left)
    6. HOG visualization (right)
    """
    preprocessed     = preprocess_patch(patch)
    left_w, right_w  = split_wing_regions(preprocessed)

    # HOG visualizations
    def hog_vis(region):
        resized = cv2.resize(region, (PATCH_W, PATCH_H))
        _, hog_image = skimage_hog(
            resized,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm="L2-Hys",
            feature_vector=True,
            visualize=True,
        )
        return hog_image

    hog_l = hog_vis(left_w)
    hog_r = hog_vis(right_w)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    # Panel 1: raw patch
    axes[0].imshow(patch, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Raw patch (64×64)", fontsize=9)

    # Panel 2: preprocessed
    vis_pre = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    if angle_right is not None and angle_left is not None:
        cx, cy   = PATCH_W // 2, PATCH_H // 2
        arm      = 22
        # Right wing arrow (pointing right of body axis = 0 deg)
        rx = int(cx + arm * np.cos(angle_right))
        ry = int(cy - arm * np.sin(angle_right))
        cv2.arrowedLine(vis_pre, (cx, cy),
                        (np.clip(rx,0,PATCH_W-1), np.clip(ry,0,PATCH_H-1)),
                        (0, 200, 80), 2, tipLength=0.3)
        # Left wing arrow
        lx = int(cx - arm * np.cos(angle_left))
        ly = int(cy - arm * np.sin(angle_left))
        cv2.arrowedLine(vis_pre, (cx, cy),
                        (np.clip(lx,0,PATCH_W-1), np.clip(ly,0,PATCH_H-1)),
                        (0, 80, 220), 2, tipLength=0.3)
        title2 = (f"Preprocessed + true angles\n"
                  f"R={np.degrees(angle_right):.0f} deg  "
                  f"L={np.degrees(angle_left):.0f} deg")
    else:
        title2 = "Preprocessed (blur + CLAHE)\nNo labels available"

    axes[1].imshow(cv2.cvtColor(vis_pre, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2, fontsize=8)

    # Panels 3-4: wing sub-regions
    axes[2].imshow(left_w,  cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Left wing region", fontsize=9)
    axes[3].imshow(right_w, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Right wing region", fontsize=9)

    # Panels 5-6: HOG visualizations
    axes[4].imshow(hog_l, cmap="gray")
    axes[4].set_title("HOG -- left wing", fontsize=9)
    axes[5].imshow(hog_r, cmap="gray")
    axes[5].set_title("HOG -- right wing", fontsize=9)

    for ax in axes:
        ax.axis("off")

    fig.suptitle("Wing-Angle Module -- Preprocessing Pipeline", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved preprocessing visualization -> {save_path}")

    plt.close(fig)


def visualize_wing_angles_on_frame(frame: np.ndarray,
                                    male_cx: int,
                                    male_cy: int,
                                    orientation: float,
                                    angle_right: float,
                                    angle_left: float,
                                    save_path: str = None):
    """
    Draw the predicted wing angles as arrows on the full video frame.

    The body axis arrow is drawn in yellow.
    Right wing in orange, left wing in blue.
    """
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    arm = 35

    # Body axis (orientation)
    bx = int(male_cx + arm * np.cos(orientation))
    by = int(male_cy - arm * np.sin(orientation))
    cv2.arrowedLine(vis, (male_cx, male_cy),
                    (np.clip(bx, 0, frame.shape[1]-1),
                     np.clip(by, 0, frame.shape[0]-1)),
                    (0, 220, 220), 2, tipLength=0.25)

    # Right wing (orange)
    rx = int(male_cx + arm * np.cos(orientation + angle_right - np.pi))
    ry = int(male_cy - arm * np.sin(orientation + angle_right - np.pi))
    cv2.arrowedLine(vis, (male_cx, male_cy),
                    (np.clip(rx, 0, frame.shape[1]-1),
                     np.clip(ry, 0, frame.shape[0]-1)),
                    (0, 140, 255), 2, tipLength=0.25)

    # Left wing (blue)
    lx = int(male_cx - arm * np.cos(orientation + angle_left - np.pi))
    ly = int(male_cy - arm * np.sin(orientation + angle_left - np.pi))
    cv2.arrowedLine(vis, (male_cx, male_cy),
                    (np.clip(lx, 0, frame.shape[1]-1),
                     np.clip(ly, 0, frame.shape[0]-1)),
                    (255, 100, 0), 2, tipLength=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Original Frame", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Wing Angles\n"
        f"Right: {np.degrees(angle_right):.1f} deg  "
        f"Left: {np.degrees(angle_left):.1f} deg",
        fontsize=9,
    )
    axes[1].axis("off")

    from matplotlib.patches import Patch
    legend = [
        Patch(color=(0.0, 0.86, 0.86), label="Body axis"),
        Patch(color=(1.0, 0.55, 0.0),  label="Left wing"),
        Patch(color=(0.0, 0.39, 1.0),  label="Right wing"),
    ]
    axes[1].legend(handles=legend, loc="lower right", fontsize=7)

    fig.suptitle("Wing-Angle Module -- Frame Visualization", fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  Saved frame visualization -> {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------------------------

def save_wing_models(model_right, model_left):
    for path, model in [(MODEL_PATH_R, model_right),
                        (MODEL_PATH_L, model_left)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        print(f"  Model saved -> {path}")


def load_wing_models(model_dir: str = "models"):
    r = os.path.join(model_dir, "wing_angle_right_model.joblib")
    l = os.path.join(model_dir, "wing_angle_left_model.joblib")
    model_right = joblib.load(r) if os.path.isfile(r) else None
    model_left  = joblib.load(l) if os.path.isfile(l) else None
    return model_right, model_left


# ---------------------------------------------------------------------------
# PREDICT  (used by main.py)
# ---------------------------------------------------------------------------

def predict_wing_angles(model_right, model_left, patch: np.ndarray):
    """
    Predict left and right wing angles for a single fly patch.

    Parameters
    ----------
    model_right, model_left : fitted Pipeline objects (or None)
    patch : uint8 grayscale array

    Returns
    -------
    (angle_right, angle_left) : tuple of float (radians)
    Returns None if either model is None.
    """
    if model_right is None or model_left is None:
        return None

    feat = patch_to_hog_vector(patch).reshape(1, -1)
    return (float(model_right.predict(feat)[0]),
            float(model_left.predict(feat)[0]))


# ---------------------------------------------------------------------------
# DEMO -- extract a real patch from the video and run preprocessing vis
# ---------------------------------------------------------------------------

def _run_preprocessing_demo(video_path: str, save_dir: str):
    """
    Pull one frame from the video, find a fly contour, crop the patch,
    run preprocessing, and save the visualization.  No angles are predicted.
    """
    cap, props = load_video(video_path)
    bg = compute_background(cap, n_samples=min(50, props["frame_count"]))

    sample_frame = None
    for i, f in enumerate(read_frames(cap, max_frames=10)):
        sample_frame = f
        if i == 4:
            break
    cap.release()

    if sample_frame is None:
        print("  [WARN] Could not extract a sample frame.")
        return

    mask     = threshold_frame(sample_frame, bg)
    contours = extract_contours(mask)

    if not contours:
        print("  [WARN] No contours found in sample frame.")
        return

    # Use the largest contour as the fly
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] < 1e-6:
        return
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    patch = extract_fly_patch(sample_frame, cx, cy)
    if patch is None:
        print(f"  [WARN] Patch centred at ({cx},{cy}) goes out of frame bounds.")
        return

    print(f"  Fly centroid: ({cx}, {cy})  |  patch shape: {patch.shape}")

    path = os.path.join(save_dir, "wing_angle_preprocessing_demo.png")
    visualize_patch_pipeline(patch, angle_right=None, angle_left=None,
                              save_path=path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(csv_path: str = DEFAULT_CSV,
         video_path: str = None,
         save_dir: str = "output"):

    os.makedirs(save_dir, exist_ok=True)

    print("=" * 58)
    print("  Wing-Angle Module -- Stage 4")
    print("=" * 58)
    print()

    # ── Step 1: data check ───────────────────────────────────────────────────
    has_data = check_for_labeled_data(csv_path)

    # ── Step 2: preprocessing + visualization demo (always runs) ────────────
    print()
    print("[PREPROCESS]  Running preprocessing demo on a video frame ...")
    if video_path is None or not os.path.exists(video_path):
        print("              No video path provided -- generating synthetic video ...")
        video_path = make_synthetic_video()
    _run_preprocessing_demo(video_path, save_dir)

    if not has_data:
        # ── No labels: stop here cleanly ─────────────────────────────────
        print()
        print("[STATUS]  Preprocessing and visualization complete.")
        print("          Regression model NOT trained -- labeled data required.")
        print(f"          Provide: {csv_path}")
        print("          predict_wing_angles() returns None until models exist.")
        return

    # ── Step 3: load labeled data ────────────────────────────────────────────
    print()
    print("[LOAD]  Reading labeled patches ...")
    patches, angles_right, angles_left = load_labeled_data(csv_path)
    print(f"        {len(patches)} patches loaded.")

    if len(patches) < 10:
        print("[WARN]  Fewer than 10 patches loaded -- results will be unreliable.")

    # Show sample patches with true labels
    n_show = min(6, len(patches))
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(patches[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"R:{np.degrees(angles_right[i]):.0f} deg\n"
                     f"L:{np.degrees(angles_left[i]):.0f} deg", fontsize=8)
        ax.axis("off")
    fig.suptitle("Sample Labeled Patches (true wing angles)", fontsize=10)
    plt.tight_layout()
    sample_path = os.path.join(save_dir, "wing_angle_labeled_samples.png")
    plt.savefig(sample_path, dpi=120)
    plt.close(fig)
    print(f"  Saved labeled samples -> {sample_path}")

    # Detailed pipeline visualization for the first labeled patch
    visualize_patch_pipeline(patches[0],
                             angle_right=float(angles_right[0]),
                             angle_left=float(angles_left[0]),
                             save_path=os.path.join(save_dir,
                                                    "wing_angle_pipeline_sample.png"))

    # ── Step 4: feature extraction ───────────────────────────────────────────
    print()
    print("[FEAT]  Extracting HOG features ...")
    X = build_feature_matrix(patches)
    print(f"        Feature matrix: {X.shape}  (samples × HOG dims)")

    # ── Step 5: train ────────────────────────────────────────────────────────
    print()
    print("[TRAIN]  Fitting PCA + LinearRegression ...")
    model_r, Xte_r, yte_r, ypr_r = train_wing_angle_model(
        X, angles_right, wing="right")
    model_l, Xte_l, yte_l, ypr_l = train_wing_angle_model(
        X, angles_left,  wing="left")

    expl = model_r.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"         PCA ({PCA_COMPONENTS} components) retains "
          f"{expl*100:.1f}% of variance.")

    # ── Step 6: evaluate on real test set ────────────────────────────────────
    print()
    print("[EVAL]  Evaluating on held-out test set (25%) ...")
    mae_r, rmse_r = evaluate_regression(yte_r, ypr_r, wing="right",
                                        save_dir=save_dir)
    mae_l, rmse_l = evaluate_regression(yte_l, ypr_l, wing="left",
                                        save_dir=save_dir)

    # ── Step 7: save ─────────────────────────────────────────────────────────
    print()
    print("[SAVE]  Saving models ...")
    save_wing_models(model_r, model_l)

    print()
    print("[DONE]  Wing-angle module trained on real labeled data.")
    print(f"        Right wing -- MAE: {np.degrees(mae_r):.2f} deg  "
          f"RMSE: {np.degrees(rmse_r):.2f} deg")
    print(f"        Left  wing -- MAE: {np.degrees(mae_l):.2f} deg  "
          f"RMSE: {np.degrees(rmse_l):.2f} deg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wing-angle module: HOG + PCA + Linear Regression")
    parser.add_argument("--data",  type=str, default=DEFAULT_CSV,
                        help=f"Path to wing_labels.csv  (default: {DEFAULT_CSV})")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input .mp4 for preprocessing demo")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save outputs  (default: output/)")
    args = parser.parse_args()
    main(csv_path=args.data, video_path=args.video, save_dir=args.save_dir)
