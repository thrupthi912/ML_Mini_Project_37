"""
main.py
-------
Integration entry point for Real-time Detailed Video Analysis of Fruit Flies.

Frame-by-frame pipeline
-----------------------
  1. Grayscale preprocessing + background subtraction   [preprocessing.py]
  2. Otsu threshold + contour extraction                [preprocessing.py]
  3. FlyCount  -- Decision Tree (zero / one / two flies) [fly_count.py]
  4. Orientation -- image moments baseline               [orientation.py]
       (Stage 2 HOG disambiguation when labeled model present)
  5. Sex classification -- Logistic Regression           [sex_classification.py]
       (skipped with a clear overlay when model unavailable)
  6. Wing-angle regression -- HOG + PCA + LinReg         [wing_angle.py]
       (skipped with a clear overlay when model unavailable)

Display
-------
Results are drawn onto each frame and saved to output/.
Optional live OpenCV window (--display flag, requires a desktop).

FPS
---
Measured per-frame and printed to console.  The mean FPS for the whole
run is printed at the end.

Module availability
-------------------
Each module is loaded at startup.  If a trained model file is absent, the
module is marked UNAVAILABLE and the corresponding annotation is replaced
with a greyed-out "N/A" label on the output frame.  No prediction is
fabricated for an unavailable module.

Usage
-----
    python main.py                                  # synthetic video, headless
    python main.py --video input/video/test4.mp4    # real video, headless
    python main.py --video input/video/test4.mp4 --display   # live window
    python main.py --train                          # retrain FlyCount first
    python main.py --frames 20 --save_dir results/
"""

import os
import sys
import time
import argparse
import traceback

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive; safe on all systems
import matplotlib.pyplot as plt

# ── Local modules ──────────────────────────────────────────────────────────
from preprocessing import (
    make_synthetic_video, load_video, compute_background,
    read_frames, threshold_frame, extract_contours, contour_features,
)
from fly_count import (
    load_model as load_fly_count_model,
    predict_fly_count, rule_based_predict, label_to_count,
)
from orientation import (
    predict_orientation,
    load_model as load_orientation_model,
)
from sex_classification import (
    load_model as load_sex_model,
    predict_sex,
)
from wing_angle import (
    load_wing_models, predict_wing_angles, extract_fly_patch,
    PATCH_H, PATCH_W,
)


# ── Visual constants ───────────────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
CLR_ONE    = (0, 200,   0)   # green  -- single fly contour
CLR_TWO    = (0, 100, 255)   # orange -- merged pair contour
CLR_ZERO   = (140, 140, 140) # grey   -- noise contour
CLR_ORIENT = (0, 220, 220)   # yellow-cyan -- orientation arrow
CLR_WING_R = (0, 140, 255)   # orange -- right wing arrow
CLR_WING_L = (255,  80,  0)  # blue   -- left wing arrow
CLR_UNAVAIL= (80,  80,  80)  # dark grey -- unavailable module text
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (  0,   0,   0)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_models() -> dict:
    """
    Attempt to load every trained model.  Returns a status dict:

        {
          "fly_count":  {"clf": ..., "le": ..., "ok": bool},
          "orientation":{"model": ...,          "ok": bool},
          "sex":        {"model": ...,           "ok": bool},
          "wing":       {"right": ..., "left":   ..., "ok": bool},
        }

    "ok" is True only when a real trained model was loaded.
    Modules whose model file is absent get ok=False and None values.
    """
    status = {}

    # FlyCount
    fc_path = "models/fly_count_model.joblib"
    if os.path.isfile(fc_path):
        try:
            clf, le = load_fly_count_model(fc_path)
            status["fly_count"] = {"clf": clf, "le": le, "ok": True}
            print(f"  [OK]   FlyCount       << {fc_path}")
        except Exception as e:
            status["fly_count"] = {"clf": None, "le": None, "ok": False}
            print(f"  [ERR]  FlyCount load failed: {e}")
    else:
        status["fly_count"] = {"clf": None, "le": None, "ok": False}
        print(f"  [MISS] FlyCount       -- run: python fly_count.py")

    # Orientation
    orient_model = load_orientation_model()
    if orient_model is not None:
        status["orientation"] = {"model": orient_model, "ok": True}
        print(f"  [OK]   Orientation    << models/orientation_model.joblib")
    else:
        status["orientation"] = {"model": None, "ok": False}
        print(f"  [INFO] Orientation    -- using moments baseline (no flip model)")

    # Sex classification
    sex_model = load_sex_model()
    if sex_model is not None:
        status["sex"] = {"model": sex_model, "ok": True}
        print(f"  [OK]   Sex classifier << models/sex_clf_model.joblib")
    else:
        status["sex"] = {"model": None, "ok": False}
        print(f"  [MISS] Sex classifier -- needs: input/images/sex_labels.csv")

    # Wing angle
    wr, wl = load_wing_models("models")
    if wr is not None and wl is not None:
        status["wing"] = {"right": wr, "left": wl, "ok": True}
        print(f"  [OK]   Wing angle     << models/wing_angle_*.joblib")
    else:
        status["wing"] = {"right": None, "left": None, "ok": False}
        print(f"  [MISS] Wing angle     -- needs: input/images/wing_labels.csv")

    return status


def _module_summary(models: dict) -> str:
    """One-line summary string of which modules are active."""
    parts = []
    parts.append("FlyCount:" + ("DT" if models["fly_count"]["ok"] else "rule"))
    parts.append("Orient:"  + ("full" if models["orientation"]["ok"] else "moments"))
    parts.append("Sex:"     + ("LR"   if models["sex"]["ok"]         else "N/A"))
    parts.append("Wing:"    + ("LR"   if models["wing"]["ok"]        else "N/A"))
    return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# PER-FRAME PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_frame(frame: np.ndarray,
                  background: np.ndarray,
                  models: dict) -> dict:
    """
    Run the full analysis pipeline on one grayscale frame.

    Errors inside individual module calls are caught and reported per
    contour so one bad contour never aborts the whole frame.

    Returns
    -------
    dict with keys:
        contours    : list of OpenCV contours
        labels      : list of str  ("zero"/"one"/"two") -- fly count per contour
        counts      : list of int  (0/1/2)
        orientations: list of float or None  (radians)
        sexes       : list of str or None    ("male"/"female"/"unknown")
        wing_info   : dict {contour_idx: {"right": float, "left": float}}
        total_flies : int
        errors      : list of str  (non-fatal errors encountered)
    """
    mask     = threshold_frame(frame, background)
    contours = extract_contours(mask)

    labels, counts, orientations, sexes = [], [], [], []
    wing_info = {}
    errors    = []

    fc   = models["fly_count"]
    ori  = models["orientation"]
    sex  = models["sex"]
    wing = models["wing"]

    for i, contour in enumerate(contours):

        # ── Fly count ───────────────────────────────────────────────────────
        try:
            if fc["ok"]:
                label = predict_fly_count(fc["clf"], fc["le"], contour)
            else:
                label = rule_based_predict(contour)
        except Exception as e:
            label = "zero"
            errors.append(f"FlyCount contour {i}: {e}")

        labels.append(label)
        counts.append(label_to_count(label))

        # ── Centroid + patch ────────────────────────────────────────────────
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

        patch = extract_fly_patch(frame, cx, cy)

        # ── Orientation ─────────────────────────────────────────────────────
        if patch is not None:
            try:
                angle = predict_orientation(patch,
                                            model=ori["model"] if ori["ok"] else None)
                orientations.append(angle)
            except Exception as e:
                orientations.append(None)
                errors.append(f"Orientation contour {i}: {e}")
        else:
            orientations.append(None)

        # ── Sex classification ───────────────────────────────────────────────
        if sex["ok"] and patch is not None:
            try:
                sex_label, _ = predict_sex(sex["model"], contour)
                sexes.append(sex_label)
            except Exception as e:
                sexes.append("unknown")
                errors.append(f"Sex contour {i}: {e}")
        else:
            sexes.append(None)   # None = module unavailable (not "unknown")

        # ── Wing angles ──────────────────────────────────────────────────────
        if label == "one" and patch is not None and wing["ok"]:
            try:
                result = predict_wing_angles(wing["right"], wing["left"], patch)
                if result is not None:
                    wing_info[i] = {"right": result[0], "left": result[1]}
            except Exception as e:
                errors.append(f"Wing contour {i}: {e}")

    total = sum(counts)
    return {
        "contours":     contours,
        "labels":       labels,
        "counts":       counts,
        "orientations": orientations,
        "sexes":        sexes,
        "wing_info":    wing_info,
        "total_flies":  total,
        "errors":       errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FRAME ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════

def _put_text_bg(img, text, org, font_scale=0.38, thickness=1,
                 fg=CLR_WHITE, bg=CLR_BLACK):
    """Draw text with a black background rectangle for readability."""
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 1, y - th - 3), (x + tw + 1, y + bl), bg, -1)
    cv2.putText(img, text, (x, y), FONT, font_scale, fg, thickness, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray,
                   result: dict,
                   models: dict,
                   fps: float = 0.0) -> np.ndarray:
    """
    Draw all available predictions onto a BGR colour copy of the frame.

    Annotations per contour
    -----------------------
    - Coloured outline (green=one, orange=two, grey=zero)
    - Fly-count label + area in pixels
    - Orientation arrow (cyan)  -- always drawn from moments baseline or model
    - Sex label (orange=male, blue=female) -- drawn only when model is available
    - Wing angle arrows (orange=right, blue=left) -- drawn only when available

    HUD (top-left corner)
    ---------------------
    - FPS counter
    - Total fly count
    - One line per module showing its status (active / unavailable)
    """
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    colour_map = {"zero": CLR_ZERO, "one": CLR_ONE, "two": CLR_TWO}

    for i, (contour, label) in enumerate(zip(result["contours"],
                                              result["labels"])):
        clr = colour_map.get(label, CLR_WHITE)
        cv2.drawContours(out, [contour], -1, clr, 2)

        M = cv2.moments(contour)
        if M["m00"] < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        area = int(cv2.contourArea(contour))

        # Fly-count label
        _put_text_bg(out, f"{label}({area})", (cx - 2, cy - 14),
                     fg=clr, bg=CLR_BLACK)

        # ── Orientation arrow ───────────────────────────────────────────────
        ang = result["orientations"][i] if i < len(result["orientations"]) else None
        if ang is not None:
            arr = 22
            tx = int(cx + arr * np.cos(ang))
            ty = int(cy - arr * np.sin(ang))
            tx = np.clip(tx, 0, out.shape[1] - 1)
            ty = np.clip(ty, 0, out.shape[0] - 1)
            cv2.arrowedLine(out, (cx, cy), (tx, ty),
                            CLR_ORIENT, 2, tipLength=0.35)

        # ── Sex label ───────────────────────────────────────────────────────
        sex_val = result["sexes"][i] if i < len(result["sexes"]) else None
        if sex_val is not None:
            sex_clr = (0, 140, 255) if sex_val == "male" else (255, 60, 0)
            _put_text_bg(out, sex_val[0].upper(),   # "M" or "F"
                         (cx + 16, cy - 14), fg=sex_clr)
        elif not models["sex"]["ok"]:
            _put_text_bg(out, "sex:N/A", (cx + 4, cy - 14), fg=CLR_UNAVAIL)

        # ── Wing angle arrows ───────────────────────────────────────────────
        if i in result["wing_info"]:
            wi  = result["wing_info"][i]
            arm = 28
            if ang is not None:
                # Right wing
                ra  = ang + wi["right"] - np.pi
                rxe = int(cx + arm * np.cos(ra))
                rye = int(cy - arm * np.sin(ra))
                cv2.arrowedLine(out, (cx, cy),
                                (np.clip(rxe, 0, out.shape[1]-1),
                                 np.clip(rye, 0, out.shape[0]-1)),
                                CLR_WING_R, 1, tipLength=0.3)
                # Left wing
                la  = ang - wi["left"] - np.pi
                lxe = int(cx + arm * np.cos(la))
                lye = int(cy - arm * np.sin(la))
                cv2.arrowedLine(out, (cx, cy),
                                (np.clip(lxe, 0, out.shape[1]-1),
                                 np.clip(lye, 0, out.shape[0]-1)),
                                CLR_WING_L, 1, tipLength=0.3)
            # Text below centroid
            wr_deg = np.degrees(wi["right"])
            wl_deg = np.degrees(wi["left"])
            _put_text_bg(out, f"R:{wr_deg:.0f} L:{wl_deg:.0f}",
                         (cx - 20, cy + 20), fg=(180, 100, 255))
        elif label == "one" and not models["wing"]["ok"]:
            _put_text_bg(out, "wing:N/A", (cx - 20, cy + 20), fg=CLR_UNAVAIL)

    # ── HUD ─────────────────────────────────────────────────────────────────
    hud_lines = [
        f"FPS: {fps:.1f}",
        f"Flies: {result['total_flies']}",
        "",
        f"FlyCount : {'DT' if models['fly_count']['ok'] else 'rule-based'}",
        f"Orient   : {'model+moments' if models['orientation']['ok'] else 'moments only'}",
        f"Sex      : {'logistic reg.' if models['sex']['ok'] else '-- unavailable --'}",
        f"Wing     : {'HOG+PCA+LR' if models['wing']['ok'] else '-- unavailable --'}",
    ]
    y0 = 16
    for line in hud_lines:
        if not line:
            y0 += 6
            continue
        fg = CLR_WHITE if ("unavailable" not in line) else CLR_UNAVAIL
        _put_text_bg(out, line, (6, y0), font_scale=0.36, fg=fg)
        y0 += 15

    # ── Error indicator ──────────────────────────────────────────────────────
    if result["errors"]:
        _put_text_bg(out, f"warn:{len(result['errors'])}",
                     (out.shape[1] - 80, 16), font_scale=0.36,
                     fg=(0, 80, 255))

    return out


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY FIGURE  (matplotlib, saved to disk)
# ═══════════════════════════════════════════════════════════════════════════

def save_summary_figure(annotated_bgr_frames: list,
                        results: list,
                        save_path: str):
    """Save a grid of annotated frames as a single PNG."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    n   = len(annotated_bgr_frames)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, bgr, res in zip(axes, annotated_bgr_frames, results):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"flies={res['total_flies']}  "
            f"contours={len(res['contours'])}",
            fontsize=8,
        )
        ax.axis("off")
    plt.suptitle("Fruit Fly Pipeline -- Annotated Frames", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary figure -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CONSOLE TABLE
# ═══════════════════════════════════════════════════════════════════════════

def _print_frame_result(frame_idx: int, result: dict, fps: float):
    """Print a compact per-frame result table to stdout."""
    print(f"\n  Frame {frame_idx:04d}  |  "
          f"flies={result['total_flies']}  "
          f"contours={len(result['contours'])}  "
          f"fps={fps:.1f}")

    for i, (lbl, cnt, ang, sx) in enumerate(zip(
            result["labels"], result["counts"],
            result["orientations"], result["sexes"])):

        ang_s = f"{np.degrees(ang):+6.1f} deg" if ang is not None else "  N/A  "
        sx_s  = sx if sx is not None else "N/A"
        wi    = result["wing_info"].get(i)
        wing_s = (f"R{np.degrees(wi['right']):+.0f} deg L{np.degrees(wi['left']):+.0f} deg"
                  if wi else "")

        print(f"    [{i}] {lbl:4s} n={cnt}  "
              f"orient={ang_s}  sex={sx_s:7s}  {wing_s}")

    if result["errors"]:
        for e in result["errors"]:
            print(f"    WARN: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def train_fly_count():
    """Train (or retrain) the FlyCount Decision Tree on synthetic data."""
    import fly_count
    fly_count.main(data_csv=None, save_dir="output")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(video_path: str = None,
         do_train: bool = False,
         max_frames: int = 10,
         save_dir: str = "output",
         display: bool = False):
    """
    Full pipeline runner.

    Parameters
    ----------
    video_path  : path to an .mp4 file.  None -> synthetic demo video.
    do_train    : if True, retrain FlyCount before running.
    max_frames  : number of frames to process.
    save_dir    : directory for all output files.
    display     : if True, show a live OpenCV window (needs a desktop).
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 62)
    print("  Fruit Fly Video Analysis -- Group 37")
    print("=" * 62)

    # ── Optional retrain ────────────────────────────────────────────────────
    if do_train:
        print("\n[TRAIN] Retraining FlyCount model ...")
        try:
            train_fly_count()
        except Exception as e:
            print(f"  [WARN] Training failed: {e}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[LOAD]  Loading models ...")
    models = load_all_models()

    # Auto-train FlyCount if its model is missing
    if not models["fly_count"]["ok"]:
        print("\n[AUTO]  FlyCount model missing -- training on synthetic data ...")
        try:
            train_fly_count()
            clf, le = load_fly_count_model("models/fly_count_model.joblib")
            models["fly_count"] = {"clf": clf, "le": le, "ok": True}
            print("  [OK]   FlyCount trained and loaded.")
        except Exception as e:
            print(f"  [WARN] Auto-train failed ({e}). Rule-based fallback active.")

    print(f"\n  Module summary: {_module_summary(models)}")

    # ── Video source ─────────────────────────────────────────────────────────
    print()
    if video_path is None or not os.path.isfile(video_path):
        if video_path is not None:
            print(f"[WARN]  Video not found: {video_path}")
        print("[VIDEO] Generating synthetic demo video ...")
        video_path = make_synthetic_video()

    print(f"[VIDEO] {video_path}")

    try:
        cap, props = load_video(video_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Cannot open video: {e}")
        sys.exit(1)

    print(f"        {props['width']}×{props['height']} @ "
          f"{props['fps']:.1f} fps  |  "
          f"{props['frame_count']} total frames  |  "
          f"processing {max_frames}")

    # ── Background model ─────────────────────────────────────────────────────
    print("\n[BG]    Computing background model ...")
    n_bg = min(50, props["frame_count"])
    try:
        background = compute_background(cap, n_samples=n_bg)
    except Exception as e:
        print(f"  [WARN] Background computation failed ({e}). Using None.")
        background = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Live window setup ────────────────────────────────────────────────────
    WIN = "Fruit Fly Analysis  [q = quit]"
    if display:
        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 960, 480)
        except Exception:
            display = False
            print("  [WARN] OpenCV display unavailable -- headless mode.")

    # ── Frame loop ───────────────────────────────────────────────────────────
    print(f"\n[RUN]   Processing frames ...\n")

    annotated_frames = []
    all_results      = []
    fps_log          = []

    frame_gen = read_frames(cap, max_frames=max_frames)
    frame_idx = 0

    for frame in frame_gen:
        t_start = time.perf_counter()

        # ── Analyse ─────────────────────────────────────────────────────────
        try:
            result = process_frame(frame, background, models)
        except Exception as e:
            print(f"  [ERROR] Frame {frame_idx:04d} failed: {e}")
            traceback.print_exc()
            frame_idx += 1
            continue

        t_end = time.perf_counter()
        fps   = 1.0 / max(t_end - t_start, 1e-9)
        fps_log.append(fps)

        # ── Annotate ─────────────────────────────────────────────────────────
        annotated = annotate_frame(frame, result, models, fps=fps)
        annotated_frames.append(annotated)
        all_results.append(result)

        # ── Console output ───────────────────────────────────────────────────
        _print_frame_result(frame_idx, result, fps)

        # ── Live display ─────────────────────────────────────────────────────
        if display:
            try:
                cv2.imshow(WIN, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n  [QUIT] User pressed q.")
                    break
            except Exception:
                display = False

        # ── Save individual frame ────────────────────────────────────────────
        frame_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
        try:
            cv2.imwrite(frame_path, annotated)
        except Exception as e:
            print(f"  [WARN] Could not save frame {frame_idx}: {e}")

        frame_idx += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # ── Summary ──────────────────────────────────────────────────────────────
    mean_fps = float(np.mean(fps_log)) if fps_log else 0.0
    total_flies_seen = sum(r["total_flies"] for r in all_results)

    print("\n" + "─" * 62)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Mean FPS         : {mean_fps:.2f}")
    print(f"  Total fly-frames : {total_flies_seen}")
    print(f"  Module summary   : {_module_summary(models)}")

    if annotated_frames:
        summary_path = os.path.join(save_dir, "pipeline_summary.png")
        try:
            save_summary_figure(annotated_frames, all_results, summary_path)
        except Exception as e:
            print(f"  [WARN] Could not save summary figure: {e}")

    print(f"\n  All outputs saved to: {save_dir}/")
    print("─" * 62)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fruit fly video analysis -- integrated ML pipeline (Group 37)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python main.py
      Synthetic demo (no video needed, headless output)

  python main.py --video input/video/test4.mp4
      Process a real video, save annotated frames to output/

  python main.py --video input/video/test4.mp4 --display
      Same but also open a live OpenCV preview window (needs desktop)

  python main.py --train
      Retrain the FlyCount model first, then run the demo

  python main.py --frames 30 --save_dir results/
      Process 30 frames and save to results/ instead of output/
        """,
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input .mp4 video file "
             "(omit to use an auto-generated synthetic video)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Retrain the FlyCount Decision Tree before running",
    )
    parser.add_argument(
        "--frames", type=int, default=10,
        help="Number of frames to process  (default: 10)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="output",
        help="Directory for all saved outputs  (default: output/)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show a live OpenCV preview window  "
             "(requires a desktop / display server)",
    )
    args = parser.parse_args()

    main(
        video_path=args.video,
        do_train=args.train,
        max_frames=args.frames,
        save_dir=args.save_dir,
        display=args.display,
    )
