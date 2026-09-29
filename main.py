"""
main.py
-------
Full pipeline entry point for Real-time Detailed Video Analysis of Fruit Flies.

Frame-by-frame processing order:
  1. Load video frame (grayscale)
  2. Threshold + extract contours  [preprocessing.py]
  3. Classify each contour as zero/one/two flies  [fly_count.py]
  4. Identify male vs female  [sex_classification.py]
  5. Estimate body orientation  [orientation.py]
  6. Predict wing angles for the male fly  [wing_angle.py]
  7. Print per-frame results; save an annotated summary figure.

Usage
-----
    python main.py                           # synthetic video demo
    python main.py --video input/video/test4.mp4
    python main.py --train                   # re-train models before running
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Local modules --------------------------------------------------------
from preprocessing import (make_synthetic_video, load_video,
                            compute_background, read_frames,
                            threshold_frame, extract_contours)

from fly_count import (load_model as load_fly_count_model,
                       predict_fly_count)

from wing_angle import (load_wing_models, predict_wing_angles,
                        extract_fly_patch, PATCH_H, PATCH_W)

from sex_classification import (load_model as load_sex_model,
                                 predict_sex, classify_pair)

from orientation import predict_orientation, load_model as load_orientation_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def train_all_models():
    """Re-train all models using their default (synthetic) data."""
    print("\n[TRAIN] Training FlyCount model ...")
    import fly_count
    fly_count.main()

    print("\n[TRAIN] Training WingAngle model ...")
    import wing_angle
    wing_angle.main()

    print("\n[TRAIN] (Sex classification — Member 2 module)")
    print("[TRAIN] (Orientation — geometry baseline)")


def load_models():
    """
    Load all trained models. Returns None for any model that isn't saved yet.
    """
    fly_count_clf, fly_count_le = (None, None)
    wing_model_right, wing_model_left = (None, None)
    sex_clf = None

    fc_path = "models/fly_count_model.joblib"
    if os.path.exists(fc_path):
        fly_count_clf, fly_count_le = load_fly_count_model(fc_path)
        print(f"  [OK] Loaded FlyCount model  ← {fc_path}")
    else:
        print(f"  [MISS] FlyCount model not found at {fc_path}")
        print("         Run: python fly_count.py   (or use --train flag)")

    wr_path = "models/wing_angle_right_model.joblib"
    wl_path = "models/wing_angle_left_model.joblib"
    if os.path.exists(wr_path) and os.path.exists(wl_path):
        wing_model_right, wing_model_left = load_wing_models("models")
        print(f"  [OK] Loaded WingAngle models ← models/")
    else:
        print(f"  [MISS] WingAngle models not found in models/")
        print("         Run: python wing_angle.py --data input/images/wing_labels.csv")

    sex_clf = load_sex_model()
    if sex_clf:
        print("  [OK] Loaded Sex classifier   ← models/sex_clf_model.joblib")
    else:
        print("  [MISS] Sex classifier not found (Member 2 module)")

    orient_model = load_orientation_model()
    if orient_model:
        print("  [OK] Loaded Orientation model ← models/orientation_model.joblib")
    else:
        print("  [MISS] Orientation model not found — using moments baseline")

    return fly_count_clf, fly_count_le, wing_model_right, wing_model_left, sex_clf, orient_model


def annotate_frame(frame_bgr: np.ndarray,
                   contours: list,
                   labels: list,
                   orientations: list,
                   wing_info: dict) -> np.ndarray:
    """
    Draw contours, labels, orientations, and wing angles on a colour frame.

    Parameters
    ----------
    frame_bgr   : colour copy of the frame (uint8 BGR)
    contours    : list of OpenCV contours
    labels      : list of str ("zero"/"one"/"two") per contour
    orientations: list of float (radians) per contour
    wing_info   : dict {contour_index: {"right": angle, "left": angle}}

    Returns
    -------
    annotated : BGR uint8
    """
    colour_map = {"zero": (128, 128, 128),   # grey
                  "one":  (0, 200, 0),        # green
                  "two":  (0, 80, 255)}        # orange

    for i, (contour, label) in enumerate(zip(contours, labels)):
        colour = colour_map.get(label, (255, 255, 255))
        cv2.drawContours(frame_bgr, [contour], -1, colour, 2)

        # Label text near centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame_bgr, label, (cx - 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

            # Orientation arrow
            if i < len(orientations) and orientations[i] is not None:
                ang = orientations[i]
                arrow_len = 20
                tip_x = int(cx + arrow_len * np.cos(ang))
                tip_y = int(cy - arrow_len * np.sin(ang))
                cv2.arrowedLine(frame_bgr, (cx, cy), (tip_x, tip_y),
                                (255, 255, 0), 1, tipLength=0.3)

            # Wing angles
            if i in wing_info:
                wr = wing_info[i].get("right")
                wl = wing_info[i].get("left")
                text = ""
                if wr is not None:
                    text += f"R:{np.degrees(wr):.0f}d "
                if wl is not None:
                    text += f"L:{np.degrees(wl):.0f}d"
                cv2.putText(frame_bgr, text.strip(),
                            (cx - 20, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 255), 1)

    return frame_bgr


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def process_frame(frame: np.ndarray,
                  background: np.ndarray,
                  fly_count_clf, fly_count_le,
                  wing_model_right, wing_model_left,
                  sex_clf, orient_model=None) -> dict:
    """
    Run the full analysis pipeline on one grayscale frame.

    Returns
    -------
    result : dict with keys:
        contours, labels, sexes, orientations, wing_info, total_flies
    """
    from preprocessing import extract_contours, contour_features

    mask     = threshold_frame(frame, background)
    contours = extract_contours(mask)

    labels       = []
    sexes        = []
    orientations = []
    wing_info    = {}
    total_flies  = 0

    for i, contour in enumerate(contours):
        # --- Fly count label -------------------------------------------------
        if fly_count_clf is not None:
            label = predict_fly_count(fly_count_clf, fly_count_le, contour)
        else:
            # Heuristic fallback
            area = contour_features(contour)["area"]
            label = "zero" if area < 80 else ("one" if area < 420 else "two")
        labels.append(label)

        n_flies_in_contour = {"zero": 0, "one": 1, "two": 2}.get(label, 0)
        total_flies += n_flies_in_contour

        # --- Crop patch for this contour -----------------------------------
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = frame.shape[1] // 2, frame.shape[0] // 2

        patch = extract_fly_patch(frame, cx, cy)

        # --- Orientation ----------------------------------------------------
        if patch is not None:
            angle = predict_orientation(patch, model=orient_model)
            orientations.append(angle)
        else:
            orientations.append(None)

        # --- Sex (placeholder) ----------------------------------------------
        if sex_clf is not None and patch is not None:
            sex, _ = predict_sex(sex_clf, contour)
        else:
            sex = "unknown"
        sexes.append(sex)

        # --- Wing angle (only if label is "one" — single male fly) ---------
        if label == "one" and patch is not None:
            if wing_model_right is not None and wing_model_left is not None:
                try:
                    result = predict_wing_angles(wing_model_right,
                                                 wing_model_left, patch)
                    if result is not None:
                        wr, wl = result
                        wing_info[i] = {"right": wr, "left": wl}
                except Exception:
                    pass   # patch too small or other issue — skip silently

    return {
        "contours":    contours,
        "labels":      labels,
        "sexes":       sexes,
        "orientations": orientations,
        "wing_info":   wing_info,
        "total_flies": total_flies,
    }


# ---------------------------------------------------------------------------
# Summary figure
# ---------------------------------------------------------------------------

def save_summary_figure(frames_rgb: list,
                        results_list: list,
                        save_path: str = "output/pipeline_summary.png"):
    """
    Save a grid of annotated frames as a single summary figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(frames_rgb)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, frame_rgb, res in zip(axes, frames_rgb, results_list):
        ax.imshow(frame_rgb)
        ax.set_title(f"Flies detected: {res['total_flies']}\n"
                     f"Labels: {res['labels']}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Fruit Fly Pipeline — Annotated Frames", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary figure saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(video_path: str = None,
         do_train: bool = False,
         max_frames: int = 5,
         save_dir: str = "output"):

    print("=" * 60)
    print("  Fruit Fly Video Analysis — Full Pipeline")
    print("=" * 60)

    # ---- Optionally re-train ------------------------------------------------
    if do_train:
        train_all_models()

    # ---- Load models --------------------------------------------------------
    print("\n[LOAD] Loading trained models ...")
    (fly_count_clf, fly_count_le,
     wing_model_right, wing_model_left,
     sex_clf, orient_model) = load_models()

    # ---- Auto-train if models are missing -----------------------------------
    if fly_count_clf is None or wing_model_right is None:
        print("\n[AUTO] Some models missing — running auto-train with synthetic data ...")
        train_all_models()
        (fly_count_clf, fly_count_le,
         wing_model_right, wing_model_left,
         sex_clf, orient_model) = load_models()

    # ---- Video source -------------------------------------------------------
    if video_path is None or not os.path.exists(video_path):
        print("\n[VIDEO] No real video found — generating synthetic demo ...")
        video_path = make_synthetic_video()
    else:
        print(f"\n[VIDEO] Using: {video_path}")

    cap, props = load_video(video_path)
    print(f"        {props['width']}x{props['height']} @ "
          f"{props['fps']:.1f} fps, {props['frame_count']} frames")

    # ---- Background model ---------------------------------------------------
    print("\n[BG]   Computing background model ...")
    background = compute_background(cap, n_samples=min(50, props["frame_count"]))

    # ---- Process frames -----------------------------------------------------
    print(f"\n[RUN]  Processing first {max_frames} frames ...\n")
    annotated_frames = []
    results_list     = []

    for frame_idx, frame in enumerate(read_frames(cap, max_frames=max_frames)):
        result = process_frame(
            frame, background,
            fly_count_clf, fly_count_le,
            wing_model_right, wing_model_left,
            sex_clf, orient_model
        )

        # Build annotated colour image for saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        annotated = annotate_frame(
            frame_bgr.copy(),
            result["contours"],
            result["labels"],
            result["orientations"],
            result["wing_info"]
        )
        annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        results_list.append(result)

        # Console output
        print(f"  Frame {frame_idx:03d}: "
              f"{len(result['contours'])} contour(s), "
              f"total flies = {result['total_flies']}")

        for j, (lbl, sx, ang) in enumerate(zip(result["labels"],
                                                result["sexes"],
                                                result["orientations"])):
            orient_str = (f"{np.degrees(ang):.1f}°"
                          if ang is not None else "N/A")
            print(f"           Contour {j}: {lbl}, sex={sx}, "
                  f"orientation={orient_str}")
            if j in result["wing_info"]:
                wi = result["wing_info"][j]
                print(f"                      wing R={np.degrees(wi.get('right', 0)):.1f}°  "
                      f"L={np.degrees(wi.get('left', 0)):.1f}°")

    cap.release()

    # ---- Save summary -------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    save_summary_figure(annotated_frames, results_list,
                        save_path=os.path.join(save_dir, "pipeline_summary.png"))

    print("\n[DONE] Pipeline complete. Check the output/ folder for results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fruit fly video analysis — full ML pipeline")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input .mp4 video")
    parser.add_argument("--train", action="store_true",
                        help="Re-train all models before running the pipeline")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of frames to process (default: 5)")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save outputs")
    args = parser.parse_args()

    main(video_path=args.video,
         do_train=args.train,
         max_frames=args.frames,
         save_dir=args.save_dir)
