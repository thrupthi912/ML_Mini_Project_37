"""
preprocessing.py  (revised)
---------------------------
Stage 1 -- Video loading, arena detection, illumination-corrected
thresholding, contour extraction, and visualization.

What changed vs. the first version
----------------------------------
1. Background is a MEDIAN of frames sampled across the whole video (not a
   mean of the first 50). A mean bakes the flies' motion trail into the
   background as a "ghost", which then shows up as a long white smear.
2. The background is passed through a large morphological closing so that any
   fly that stayed still is removed, leaving only the illumination profile
   (bright floor + dark vignetted wall).
3. The circular arena is detected once and everything outside a shrunken copy
   of it is ignored, so wall shadows can't leak into the mask.
4. Otsu is computed only over arena pixels (the dark wall ring used to skew it)
   and has a floor (`min_diff`) so an empty frame doesn't threshold noise.
5. Area limits scale with frame size instead of being fixed at 50-5000 px.

Works standalone:
    python preprocessing.py                          # synthetic video
    python preprocessing.py --video path/to/file.mp4
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video(path: str):
    """Open a video and return (cap, props)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    props = {
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, props


def _to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def read_frames(cap, max_frames: int = None):
    """Generator -- yields one grayscale frame at a time."""
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield _to_gray(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break


# ---------------------------------------------------------------------------
# Illumination model + arena detection
# ---------------------------------------------------------------------------

def estimate_illumination(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Estimate the fly-free illumination of a grayscale image.

    Flies are dark and small relative to the arena, so a grayscale CLOSING with
    a kernel larger than a fly erases them and keeps the large-scale shading
    (bright floor, dark wall). Done at 1/scale resolution for speed.
    """
    h, w = img.shape
    small = cv2.resize(img, (max(1, w // scale), max(1, h // scale)),
                       interpolation=cv2.INTER_AREA)
    k = int(0.18 * min(h, w) / scale) | 1            # odd; ~0.18 x frame side
    k = max(k, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    return cv2.resize(closed, (w, h), interpolation=cv2.INTER_LINEAR)


def compute_background(cap, n_samples: int = 30) -> np.ndarray:
    """
    Median of `n_samples` frames spread evenly over the whole video, cleaned by
    `estimate_illumination`. Rewinds the capture before returning.

    Returns
    -------
    illumination : np.ndarray uint8 -- what an empty arena looks like.
    """
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_samples, max(n_total, 1))
    idxs = np.linspace(0, max(n_total - 1, 0), n).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(_to_gray(frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)               # rewind

    if not frames:
        raise RuntimeError("Could not read any frames to build a background.")

    median_bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return estimate_illumination(median_bg)


def detect_arena(illum: np.ndarray):
    """
    Find the bright circular arena floor in the illumination image.

    Returns (cx, cy, r) in pixels, or None if the whole frame is bright
    (e.g. the synthetic demo), in which case no arena mask is applied.
    """
    h, w = illum.shape
    blur = cv2.GaussianBlur(illum, (0, 0), max(h, w) / 200.0)
    if int(blur.max()) - int(blur.min()) < 40:
        return None                                    # flat image: no arena
    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(big)
    if area > 0.95 * h * w or area < 0.1 * h * w:
        return None                                    # no clear arena

    m = cv2.moments(big)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    r = float(np.sqrt(area / np.pi))                   # robust to ragged edge
    return cx, cy, r


def arena_mask(shape, arena, shrink: float = 0.94) -> np.ndarray:
    """Filled circle (255 inside). `shrink` keeps us clear of the wall."""
    mask = np.zeros(shape, np.uint8)
    if arena is None:
        mask[:] = 255
    else:
        cx, cy, r = arena
        cv2.circle(mask, (int(round(cx)), int(round(cy))),
                   int(r * shrink), 255, -1)
    return mask


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def threshold_frame(frame: np.ndarray,
                    background: np.ndarray = None,
                    blur_ksize: int = 5,
                    arena=None,
                    min_diff: int = 20) -> np.ndarray:
    """
    Binary mask of fly pixels (255 = fly).

    1. darkness = illumination - frame   (flies are darker than the floor)
       `background` is the illumination map from compute_background(); if None
       it is estimated from this single frame.
    2. Gaussian blur.
    3. Otsu threshold computed over ARENA pixels only, floored at `min_diff`.
    4. Opening removes specks; closing bridges wing/body gaps.
    """
    illum = background if background is not None else estimate_illumination(frame)
    darkness = cv2.subtract(illum, frame)              # saturating: >= 0
    blurred = cv2.GaussianBlur(darkness, (blur_ksize, blur_ksize), 0)

    inside = arena_mask(frame.shape, arena)
    vals = blurred[inside > 0].reshape(-1, 1)
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(t, min_diff)

    mask = np.where((blurred > t) & (inside > 0), 255, 0).astype(np.uint8)

    # Morphology kernels scale with resolution (5 px at ~1500 px, 3 px at 320)
    k = max(3, int(round(min(frame.shape) / 300))) | 1
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    return mask


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

# Fly area as a fraction of frame area. Estimated from the 1530x1530 sample
# frames (a fly with wings is ~10-13k px there). Override with --min_area /
# --max_area once you've looked at the printed areas.
MIN_AREA_FRAC = 0.0006
MAX_AREA_FRAC = 0.012


def extract_contours(mask: np.ndarray,
                     min_area: float = None,
                     max_area: float = None):
    """
    External contours filtered by area. If limits are None they are derived
    from the frame size (see MIN_AREA_FRAC / MAX_AREA_FRAC).
    """
    frame_area = mask.shape[0] * mask.shape[1]
    if min_area is None:
        min_area = MIN_AREA_FRAC * frame_area
    if max_area is None:
        max_area = MAX_AREA_FRAC * frame_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def contour_features(contour) -> dict:
    """
    Geometric features used by the FlyCount classifier:
    area, perimeter, aspect_ratio, extent, solidity.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = area / float(w * h) if (w * h) > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area > 0 else 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_frame(frame, mask, contours, title="Frame",
                    save_path=None, arena=None):
    """3-panel figure: frame | mask | frame + contours (green) + arena (blue)."""
    contour_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    lw = max(2, min(frame.shape) // 400)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), lw)
    if arena is not None:
        cx, cy, r = arena
        cv2.circle(contour_img, (int(cx), int(cy)), int(r * 0.94),
                   (255, 0, 0), lw)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Grayscale Frame")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask (illumination-corrected Otsu)")
    axes[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Contours detected: {len(contours)}")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved visualization -> {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Synthetic video generator (for demo / testing without real data)
# ---------------------------------------------------------------------------

def make_synthetic_video(path: str = "synthetic_test.mp4",
                         n_frames: int = 60,
                         width: int = 320,
                         height: int = 240):
    """Two moving dark ellipses on a light, noisy background."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (width, height), isColor=False)
    rng = np.random.default_rng(42)

    pos = np.array([[80, 120], [240, 100]], dtype=float)
    vel = np.array([[2.0, 1.5], [-1.8, 2.2]])

    for _ in range(n_frames):
        frame = np.full((height, width), 230, dtype=np.uint8)
        for i in range(2):
            cx, cy = int(pos[i, 0]), int(pos[i, 1])
            cv2.ellipse(frame, (cx, cy), (12, 7), 0, 0, 360, 30, -1)
            pos[i] += vel[i]
            if pos[i, 0] < 20 or pos[i, 0] > width - 20:
                vel[i, 0] *= -1
            if pos[i, 1] < 20 or pos[i, 1] > height - 20:
                vel[i, 1] *= -1

        noise = rng.integers(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"  Synthetic video saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_preprocessing(video_path: str,
                      max_frames: int = 5,
                      save_dir: str = "output",
                      min_area: float = None,
                      max_area: float = None):
    """
    1. Load video  2. Build illumination model + detect arena
    3. Threshold + extract contours for the first `max_frames` frames
    4. Visualize and print contour features

    Returns list of (frame, mask, contours) tuples.
    """
    print(f"\n=== Preprocessing: {video_path} ===")
    cap, props = load_video(video_path)
    print(f"  Video: {props['width']}x{props['height']} @ {props['fps']:.1f} fps, "
          f"{props['frame_count']} frames")

    print("  Building illumination model (median over whole video)...")
    background = compute_background(cap)
    arena = detect_arena(background)
    if arena is None:
        print("  Arena: not detected -> using full frame")
    else:
        print(f"  Arena: centre=({arena[0]:.0f}, {arena[1]:.0f}), r={arena[2]:.0f}px")

    results = []
    for frame_idx, frame in enumerate(read_frames(cap, max_frames=max_frames)):
        mask = threshold_frame(frame, background, arena=arena)
        contours = extract_contours(mask, min_area, max_area)

        print(f"\n  Frame {frame_idx:03d}: {len(contours)} contour(s) detected")
        for j, c in enumerate(contours):
            f = contour_features(c)
            print(f"    Contour {j}: area={f['area']:.0f}, "
                  f"AR={f['aspect_ratio']:.2f}, solidity={f['solidity']:.2f}")

        visualize_frame(frame, mask, contours,
                        title=f"Frame {frame_idx}",
                        save_path=os.path.join(
                            save_dir, f"frame_{frame_idx:03d}_preprocessing.png"),
                        arena=arena)
        results.append((frame, mask, contours))

    cap.release()
    print("\nPreprocessing complete.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fruit fly video preprocessing: threshold + contour extraction")
    parser.add_argument("--video", type=str, default=None,
                        help="Input video. Omit for a synthetic demo.")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="output")
    parser.add_argument("--min_area", type=float, default=None,
                        help="Min contour area in px (default: scaled to frame)")
    parser.add_argument("--max_area", type=float, default=None,
                        help="Max contour area in px (default: scaled to frame)")
    args = parser.parse_args()

    if args.video is None or not os.path.exists(args.video):
        print("No real video found. Generating synthetic demo video...")
        video_path = make_synthetic_video()
    else:
        video_path = args.video

    run_preprocessing(video_path, max_frames=args.frames, save_dir=args.save_dir,
                      min_area=args.min_area, max_area=args.max_area)