"""
preprocessing.py
----------------
Stage 1 -- Video loading, grayscale conversion, thresholding,
contour extraction, and visualization.

Works standalone:
    python preprocessing.py                          # uses synthetic video
    python preprocessing.py --video path/to/file.mp4
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- no display needed
import matplotlib.pyplot as plt
import argparse
import os


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video(path: str):
    """
    Open a video file and return the capture object plus basic properties.

    Returns
    -------
    cap : cv2.VideoCapture
    props : dict  -- fps, width, height, frame_count
    """
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


def read_frames(cap, max_frames: int = None):
    """
    Generator -- yields one grayscale frame at a time.

    Parameters
    ----------
    cap        : cv2.VideoCapture (already opened)
    max_frames : int or None -- stop after this many frames
    """
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert BGR -> grayscale if the video is colour
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        yield gray
        count += 1
        if max_frames is not None and count >= max_frames:
            break


# ---------------------------------------------------------------------------
# Background subtraction / thresholding
# ---------------------------------------------------------------------------

def compute_background(cap, n_samples: int = 50):
    """
    Estimate a static background by averaging the first `n_samples` frames.
    Rewinds the capture to frame 0 before returning.

    Returns
    -------
    background : np.ndarray  uint8 grayscale
    """
    frames = []
    for i, frame in enumerate(read_frames(cap, max_frames=n_samples)):
        frames.append(frame.astype(np.float32))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # rewind

    background = np.mean(frames, axis=0).astype(np.uint8)
    return background


def threshold_frame(frame: np.ndarray,
                    background: np.ndarray = None,
                    blur_ksize: int = 5) -> np.ndarray:
    """
    Produce a binary mask that highlights fly pixels.

    Steps
    -----
    1. If a background is given, subtract it to isolate moving objects.
    2. Gaussian blur to reduce noise.
    3. Otsu's thresholding -> binary mask.
    4. Morphological closing to fill small holes.

    Parameters
    ----------
    frame      : grayscale uint8 frame
    background : grayscale uint8 background image (optional)
    blur_ksize : kernel size for Gaussian blur (must be odd)

    Returns
    -------
    mask : binary uint8 array (0 or 255)
    """
    if background is not None:
        # Absolute difference: bright where flies are present (flies moved)
        diff = cv2.absdiff(frame, background)
        # Gaussian blur
        blurred = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
        # Otsu on the difference image -- bright pixels are foreground
        _, mask = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # No background model available.
        # Flies are dark objects on a light background, so we invert before
        # thresholding so that fly pixels become bright (foreground = high value).
        diff = frame.copy()
        blurred = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
        # THRESH_BINARY_INV: dark fly pixels -> 255, light background -> 0
        _, mask = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing: closes small dark gaps inside fly blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contours(mask: np.ndarray,
                     min_area: int = 50,
                     max_area: int = 5000):
    """
    Find external contours in a binary mask and filter by area.

    Parameters
    ----------
    mask     : binary uint8 mask
    min_area : discard contours smaller than this (noise)
    max_area : discard contours larger than this (arena boundary artefacts)

    Returns
    -------
    contours : list of np.ndarray  -- each is an (N,1,2) int32 array
    """
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours
                if min_area <= cv2.contourArea(c) <= max_area]
    return filtered


def contour_features(contour) -> dict:
    """
    Compute a set of geometric features for a single contour.
    These are the input features used by the FlyCount classifier.

    Features
    --------
    area        : pixel area of the contour
    perimeter   : arc length
    aspect_ratio: width / height of bounding rectangle
    extent      : area / bounding_rect_area
    solidity    : area / convex_hull_area
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

def visualize_frame(frame: np.ndarray,
                    mask: np.ndarray,
                    contours: list,
                    title: str = "Frame",
                    save_path: str = None):
    """
    Display (and optionally save) a 3-panel figure:
      Left  -- original grayscale frame
      Centre -- thresholded binary mask
      Right  -- original frame with contours drawn in green
    """
    contour_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Grayscale Frame")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Thresholded Mask (Otsu)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Contours detected: {len(contours)}")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    """
    Generate a simple synthetic grayscale video with two moving dark ellipses
    (simulating fruit flies) on a light background.

    Used automatically when no real video is available.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (width, height),
                             isColor=False)

    rng = np.random.default_rng(42)

    # Starting positions and velocities for two 'flies'
    pos = np.array([[80, 120], [240, 100]], dtype=float)
    vel = np.array([[2.0, 1.5], [-1.8, 2.2]])

    for _ in range(n_frames):
        frame = np.full((height, width), 230, dtype=np.uint8)  # light bg

        for i in range(2):
            cx, cy = int(pos[i, 0]), int(pos[i, 1])
            # Draw a small dark ellipse to mimic a fly
            cv2.ellipse(frame, (cx, cy), (12, 7), 0, 0, 360, 30, -1)

            # Bounce off walls
            pos[i] += vel[i]
            if pos[i, 0] < 20 or pos[i, 0] > width - 20:
                vel[i, 0] *= -1
            if pos[i, 1] < 20 or pos[i, 1] > height - 20:
                vel[i, 1] *= -1

        # Add a little Gaussian noise
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
                      save_dir: str = "output"):
    """
    Full preprocessing demo:
      1. Load video
      2. Compute background
      3. For the first `max_frames` frames: threshold + extract contours
      4. Visualize and print contour features

    Returns list of (frame, mask, contours) tuples.
    """
    print(f"\n=== Preprocessing: {video_path} ===")
    cap, props = load_video(video_path)
    print(f"  Video: {props['width']}x{props['height']} @ {props['fps']:.1f} fps, "
          f"{props['frame_count']} frames")

    print("  Computing background model...")
    background = compute_background(cap, n_samples=min(50, props["frame_count"]))

    results = []
    for frame_idx, frame in enumerate(read_frames(cap, max_frames=max_frames)):
        mask = threshold_frame(frame, background)
        contours = extract_contours(mask)

        print(f"\n  Frame {frame_idx:03d}: {len(contours)} contour(s) detected")
        for j, c in enumerate(contours):
            feats = contour_features(c)
            print(f"    Contour {j}: area={feats['area']:.1f}, "
                  f"AR={feats['aspect_ratio']:.2f}, "
                  f"solidity={feats['solidity']:.2f}")

        save_path = os.path.join(save_dir, f"frame_{frame_idx:03d}_preprocessing.png")
        visualize_frame(frame, mask, contours,
                        title=f"Frame {frame_idx}",
                        save_path=save_path)

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
                        help="Path to input .mp4 video. "
                             "Omit to auto-generate a synthetic demo video.")
    parser.add_argument("--frames", type=int, default=3,
                        help="Number of frames to visualize (default: 3)")
    parser.add_argument("--save_dir", type=str, default="output",
                        help="Directory to save visualizations")
    args = parser.parse_args()

    if args.video is None or not os.path.exists(args.video):
        print("No real video found. Generating synthetic demo video...")
        video_path = make_synthetic_video()
    else:
        video_path = args.video

    run_preprocessing(video_path, max_frames=args.frames, save_dir=args.save_dir)
