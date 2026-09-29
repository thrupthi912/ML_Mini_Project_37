"""
wing_angle.py  — Thrupthi (Person 1, Stage 4)
----------------------------------------------
Linear Regression model that predicts male fly wing angles (radians)
from HOG (Histogram of Oriented Gradients) features + PCA.

Pipeline overview
-----------------
1. For each labeled male fly patch:
   a. Locate the male fly's bounding box in the frame.
   b. Crop a fixed-size ROI centred on the fly (the "patch").
   c. Split the patch into LEFT and RIGHT wing sub-regions.
   d. Compute HOG features for each sub-region.
2. Concatenate left + right HOG vectors → one feature vector per frame.
3. Apply PCA to reduce dimensionality and remove redundancy.
4. Train separate LinearRegression models for left and right wing angles.
5. Evaluate using MAE and RMSE on a held-out test set.
6. Save model artefacts to models/.

Usage
-----
    python wing_angle.py               # synthetic demo (always works)
    python wing_angle.py --data input/images/wing_labels.csv
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import cv2

from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Constants — patch geometry
# ---------------------------------------------------------------------------

PATCH_H = 64          # height of the male fly ROI (pixels)
PATCH_W = 64          # width  of the male fly ROI (pixels)
WING_FRACTION = 0.45  # fraction of patch width dedicated to each wing region

HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS    = 9

PCA_COMPONENTS = 30   # keep top-30 principal components


# ---------------------------------------------------------------------------
# ROI / patch extraction
# ---------------------------------------------------------------------------

def extract_fly_patch(frame: np.ndarray,
                      center_x: int,
                      center_y: int,
                      patch_h: int = PATCH_H,
                      patch_w: int = PATCH_W) -> np.ndarray:
    """
    Crop a fixed-size rectangle centred on the male fly from `frame`.

    Parameters
    ----------
    frame    : grayscale uint8 frame
    center_x : estimated x-coordinate of the fly's centre
    center_y : estimated y-coordinate of the fly's centre

    Returns
    -------
    patch : uint8 array of shape (patch_h, patch_w), or None if out of bounds
    """
    h, w = frame.shape[:2]
    x0 = center_x - patch_w // 2
    y0 = center_y - patch_h // 2
    x1 = x0 + patch_w
    y1 = y0 + patch_h

    # Reject if the patch would fall outside the frame
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    patch = frame[y0:y1, x0:x1].copy()
    return patch.astype(np.uint8)


def split_wing_regions(patch: np.ndarray):
    """
    Divide the fly patch into left and right wing sub-regions.

    The male fly is assumed to be roughly centred in the patch with its body
    running vertically.  Wings appear on the left and right sides.

    Layout (patch_w = 64, WING_FRACTION = 0.45):
       columns 0 .. 28  → left wing region  (28 px wide)
       columns 36 .. 63 → right wing region (28 px wide)
       columns 29 .. 35 → body (discarded)

    Returns
    -------
    left_wing, right_wing : uint8 arrays
    """
    pw = patch.shape[1]
    wing_w = int(pw * WING_FRACTION)

    left_wing  = patch[:, :wing_w]
    right_wing = patch[:, pw - wing_w:]
    return left_wing, right_wing


# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------

def hog_features(region: np.ndarray) -> np.ndarray:
    """
    Compute HOG features for a single image region.

    HOG captures the local gradient structure — edges and their
    orientations — which is informative about wing angle and shape.

    Returns
    -------
    feat_vec : 1-D float64 array
    """
    # Resize to a fixed size to ensure consistent feature length
    region_resized = cv2.resize(region, (PATCH_W, PATCH_H))

    feat_vec = hog(region_resized,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm="L2-Hys",
                   feature_vector=True)
    return feat_vec


def patch_to_hog_vector(patch: np.ndarray) -> np.ndarray:
    """
    Extract combined HOG feature vector for a full fly patch.
    Left-wing HOG and right-wing HOG are concatenated.

    Returns
    -------
    feat_vec : 1-D float64 array  (length = 2 * hog_dim)
    """
    left_wing, right_wing = split_wing_regions(patch)
    left_hog  = hog_features(left_wing)
    right_hog = hog_features(right_wing)
    return np.concatenate([left_hog, right_hog])


# ---------------------------------------------------------------------------
# Synthetic dataset (fallback when no real labeled data exists)
# ---------------------------------------------------------------------------

def make_synthetic_patches(n_samples: int = 300,
                            random_state: int = 42):
    """
    Generate synthetic fly patches with random wing angles.

    Each patch is a 64x64 grayscale image with:
      - A dark ellipse in the centre (fly body)
      - Two dark arcs emanating from the sides (wings) at the target angle

    The labels are the angles in radians:
      angle_right : angle of the right wing from the body axis  [0, pi]
      angle_left  : angle of the left wing from the body axis   [0, pi]

    Returns
    -------
    patches        : list of np.ndarray  shape (64, 64)
    angles_right   : np.ndarray  (n_samples,)
    angles_left    : np.ndarray  (n_samples,)
    """
    rng = np.random.default_rng(random_state)
    patches = []
    angles_right = rng.uniform(0.2, np.pi - 0.2, n_samples).astype(np.float32)
    angles_left  = rng.uniform(0.2, np.pi - 0.2, n_samples).astype(np.float32)

    for i in range(n_samples):
        patch = np.full((PATCH_H, PATCH_W), 200, dtype=np.uint8)   # light bg
        cy, cx = PATCH_H // 2, PATCH_W // 2

        # Body ellipse
        cv2.ellipse(patch, (cx, cy), (8, 20), 0, 0, 360, 40, -1)

        # Right wing line
        r_angle = angles_right[i]
        wing_len = 22
        ex = int(cx + wing_len * np.cos(r_angle))
        ey = int(cy - wing_len * np.sin(r_angle))
        ex = np.clip(ex, 0, PATCH_W - 1)
        ey = np.clip(ey, 0, PATCH_H - 1)
        cv2.line(patch, (cx, cy), (ex, ey), 30, 2)

        # Left wing line
        l_angle = angles_left[i]
        ex2 = int(cx - wing_len * np.cos(l_angle))
        ey2 = int(cy - wing_len * np.sin(l_angle))
        ex2 = np.clip(ex2, 0, PATCH_W - 1)
        ey2 = np.clip(ey2, 0, PATCH_H - 1)
        cv2.line(patch, (cx, cy), (ex2, ey2), 30, 2)

        # Light noise
        noise = rng.integers(-15, 15, patch.shape, dtype=np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        patches.append(patch)

    return patches, angles_right, angles_left


# ---------------------------------------------------------------------------
# Load real labeled data
# ---------------------------------------------------------------------------

def load_labeled_patches(csv_path: str):
    """
    Load labeled wing-angle data from a CSV with columns:
        patch_path, angle_right, angle_left

    patch_path : path to a 64x64 grayscale PNG of the male fly
    angle_right: right wing angle in radians
    angle_left : left wing angle in radians

    Returns
    -------
    patches      : list of np.ndarray
    angles_right : np.ndarray
    angles_left  : np.ndarray
    """
    import csv
    patches, ar, al = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = cv2.imread(row["patch_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Could not load patch: {row['patch_path']}")
                continue
            patches.append(cv2.resize(img, (PATCH_W, PATCH_H)))
            ar.append(float(row["angle_right"]))
            al.append(float(row["angle_left"]))

    return patches, np.array(ar, dtype=np.float32), np.array(al, dtype=np.float32)


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(patches: list) -> np.ndarray:
    """
    Apply HOG feature extraction to every patch.

    Returns
    -------
    X : np.ndarray  shape (n_samples, 2 * hog_dim)
    """
    rows = [patch_to_hog_vector(p) for p in patches]
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_wing_angle_model(X: np.ndarray,
                           y: np.ndarray,
                           label_name: str = "right",
                           n_pca: int = PCA_COMPONENTS,
                           random_state: int = 42):
    """
    Train a PCA + LinearRegression pipeline for one wing angle.

    Parameters
    ----------
    X          : HOG feature matrix  (N, D)
    y          : target angles in radians  (N,)
    label_name : "right" or "left"  — used only for printing
    n_pca      : number of PCA components to retain
    random_state : seed for train_test_split

    Returns
    -------
    model   : fitted sklearn Pipeline (PCA → LinearRegression)
    X_test  : test features
    y_test  : true test labels
    y_pred  : predicted test labels
    """
    n_pca_safe = min(n_pca, X.shape[1], X.shape[0] - 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state)

    model = Pipeline([
        ("pca", PCA(n_components=n_pca_safe, random_state=random_state)),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(y_test: np.ndarray,
                        y_pred: np.ndarray,
                        label: str = "right",
                        save_dir: str = "output"):
    """
    Print MAE and RMSE; save a scatter plot of true vs predicted angles.
    """
    os.makedirs(save_dir, exist_ok=True)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  [{label.upper()} WING]")
    print(f"    MAE  = {mae:.4f} rad  ({np.degrees(mae):.2f}°)")
    print(f"    RMSE = {rmse:.4f} rad  ({np.degrees(rmse):.2f}°)")

    # Scatter: true vs predicted
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, label="Predictions")
    lims = [min(y_test.min(), y_pred.min()) - 0.1,
            max(y_test.max(), y_pred.max()) + 0.1]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("True angle (rad)")
    ax.set_ylabel("Predicted angle (rad)")
    ax.set_title(f"Wing Angle Regression — {label} wing\n"
                 f"MAE={mae:.3f} rad  RMSE={rmse:.3f} rad")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"wing_angle_{label}_scatter.png")
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"    Saved scatter plot → {plot_path}")

    return mae, rmse


def visualize_sample_patches(patches, angles_right, angles_left,
                              n: int = 6, save_dir: str = "output"):
    """
    Show n sample patches with their true wing angles annotated.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(n, len(patches))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(patches[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"R:{np.degrees(angles_right[i]):.0f}°\n"
                     f"L:{np.degrees(angles_left[i]):.0f}°",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle("Sample Fly Patches with True Wing Angles", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_wing_patches.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved sample patches → {path}")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_wing_models(model_right, model_left,
                     save_dir: str = "models"):
    """Save the two wing-angle regression pipelines."""
    os.makedirs(save_dir, exist_ok=True)
    r_path = os.path.join(save_dir, "wing_angle_right_model.joblib")
    l_path = os.path.join(save_dir, "wing_angle_left_model.joblib")
    joblib.dump(model_right, r_path)
    joblib.dump(model_left,  l_path)
    print(f"  Model saved → {r_path}")
    print(f"  Model saved → {l_path}")


def load_wing_models(model_dir: str = "models"):
    """Load previously saved wing-angle models."""
    model_right = joblib.load(os.path.join(model_dir, "wing_angle_right_model.joblib"))
    model_left  = joblib.load(os.path.join(model_dir, "wing_angle_left_model.joblib"))
    return model_right, model_left


# ---------------------------------------------------------------------------
# Prediction helper (used by main.py)
# ---------------------------------------------------------------------------

def predict_wing_angles(model_right, model_left,
                        patch: np.ndarray):
    """
    Predict left and right wing angles for a single fly patch.

    Parameters
    ----------
    model_right, model_left : fitted Pipeline objects
    patch : uint8 array shape (H, W)

    Returns
    -------
    angle_right : float (radians)
    angle_left  : float (radians)
    """
    feat_vec = patch_to_hog_vector(patch).reshape(1, -1)
    angle_right = float(model_right.predict(feat_vec)[0])
    angle_left  = float(model_left.predict(feat_vec)[0])
    return angle_right, angle_left


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_csv: str = None, save_dir: str = "output"):
    print("=" * 55)
    print("  Wing Angle Regression (HOG + PCA + LR) — Thrupthi")
    print("=" * 55)

    # ---- Data loading -------------------------------------------------------
    if data_csv and os.path.exists(data_csv):
        print(f"\n[DATA] Loading real labeled patches from: {data_csv}")
        patches, angles_right, angles_left = load_labeled_patches(data_csv)
        print(f"       Loaded {len(patches)} patches.")
    else:
        if data_csv:
            print(f"\n[WARN] CSV not found: {data_csv}")
        print("\n[DATA] No labeled patch CSV found — using SYNTHETIC patches for demo.")
        print("       To use real data, provide --data path/to/wing_labels.csv")
        print("       (see README.md for the expected CSV format)\n")
        patches, angles_right, angles_left = make_synthetic_patches(n_samples=300)
        print(f"       Generated {len(patches)} synthetic patches.")

    # ---- Visualise samples --------------------------------------------------
    visualize_sample_patches(patches, angles_right, angles_left,
                             n=6, save_dir=save_dir)

    # ---- Feature extraction -------------------------------------------------
    print("\n[FEAT] Extracting HOG features ...")
    X = build_feature_matrix(patches)
    print(f"       Feature matrix shape: {X.shape}  "
          f"(samples × HOG features)")

    # ---- Training -----------------------------------------------------------
    print("\n[TRAIN] Fitting PCA + LinearRegression ...")
    print("        Right wing ...")
    model_right, X_test_r, y_test_r, y_pred_r = train_wing_angle_model(
        X, angles_right, label_name="right")

    print("        Left wing ...")
    model_left, X_test_l, y_test_l, y_pred_l = train_wing_angle_model(
        X, angles_left, label_name="left")

    explained_var = model_right.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"\n  PCA: {PCA_COMPONENTS} components retain "
          f"{explained_var*100:.1f}% of variance.")

    # ---- Evaluation ---------------------------------------------------------
    print("\n[EVAL] Evaluating on held-out test set (25%) ...")
    mae_r, rmse_r = evaluate_regression(y_test_r, y_pred_r,
                                        label="right", save_dir=save_dir)
    mae_l, rmse_l = evaluate_regression(y_test_l, y_pred_l,
                                        label="left", save_dir=save_dir)

    # ---- Save ---------------------------------------------------------------
    print("\n[SAVE] Saving models ...")
    save_wing_models(model_right, model_left)

    print("\n[DONE] Wing angle module finished.")
    print(f"       Right wing — MAE: {np.degrees(mae_r):.2f}°  "
          f"RMSE: {np.degrees(rmse_r):.2f}°")
    print(f"       Left  wing — MAE: {np.degrees(mae_l):.2f}°  "
          f"RMSE: {np.degrees(rmse_l):.2f}°")

    if data_csv is None or not os.path.exists(data_csv):
        print("\n[NOTE] Results above are on SYNTHETIC data.")
        print("       Real errors will differ once real labeled patches are used.")
        print("       Download labeled data — see README.md for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate wing angle regression model")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to wing_labels.csv "
                             "(omit to run on synthetic demo data)")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save outputs (default: output/)")
    args = parser.parse_args()

    main(data_csv=args.data, save_dir=args.save_dir)
