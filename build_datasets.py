#!/usr/bin/env python3
"""
build_datasets.py - build validator CSVs in input/images/ from the LOCAL
extracted fruit-fly dataset (LabelMe point JSONs + images).
Run from the repo root:  python build_datasets.py

Deterministic and local. No values are invented; if something cannot be
derived from the files, the script prints the ambiguity and exits non-zero.
Requires: numpy, opencv-python

Derivations
  fly_count_labels.csv : blob features from the Otsu-segmented image contour;
                         label = "zero"/"one"/"two" by number of head points
                         (fh, mh) inside the blob. Blobs with 3+ heads are skipped.
  sex_labels.csv       : blobs with exactly one head point; fh -> "female", mh -> "male".
  orientation_labels.csv: same single-fly blobs; moment_angle_rad = 0.5*atan2(2*mu11, mu20-mu02)
                         (image coords, y down); flip = 1 if the head lies on the negative
                         side of that axis (dot(head-centroid, (cos,sin)) < 0), else 0.
                         patch_path is relative to the repo root.
  wing_labels.csv      : single-male blobs (one 'mh' head point) with exactly two 'mw' points.
                         Convention taken from wing_angle.py: origin = blob centroid; the
                         angle is the unsigned angle in [0, pi] between the origin->wing-tip
                         vector and the POSTERIOR body axis (centroid->head reversed);
                         right/left = the fly's own side in dorsal view (head up => right is
                         image-right). patch_path = 64x64 grayscale PNG centred on the
                         centroid and rotated so the head points up (real pixels only).
                         Blobs whose sides are ambiguous are skipped and reported.
                         Never written header-only.
"""
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

IMAGES_DIR = Path("input/images")
PATCH_DIR = IMAGES_DIR / "patches"
PATCH_SIZE = 128
MIN_AREA = 200.0
OPEN_KERNEL = 3
MIN_ELONGATION = 1.1
WING_PATCH = 64  # wing_angle.py expects 64x64 grayscale patches

KNOWN_LABELS = {"fh", "fp", "fa", "mh", "mp", "ma", "mp2", "mw"}
HEAD_SEX = {"fh": "female", "mh": "male"}
COUNT_NAME = {0: "zero", 1: "one", 2: "two"}
FEATS = ["area", "perimeter", "aspect_ratio", "extent", "solidity"]
COLS = {
    "fly_count_labels.csv": FEATS + ["label"],
    "sex_labels.csv": FEATS + ["label"],
    "orientation_labels.csv": ["patch_path", "moment_angle_rad", "flip"],
    "wing_labels.csv": ["patch_path", "angle_right_rad", "angle_left_rad"],
}

skipped = Counter()


def die(msg):
    print(f"\nERROR (stopping, nothing guessed): {msg}", file=sys.stderr)
    sys.exit(2)


def write_csv(name, rows):
    path = IMAGES_DIR / name
    fd, tmp = tempfile.mkstemp(dir=IMAGES_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLS[name])
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_image(jpath, data):
    ip = data.get("imagePath")
    if ip:
        cand = jpath.parent / str(ip).replace("\\", "/")
        if cand.is_file():
            return cv2.imread(str(cand), cv2.IMREAD_COLOR)
    return None  # only files on disk are used


def features(cnt):
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    if area <= 0 or w == 0 or h == 0 or hull <= 0:
        return None
    return [area, cv2.arcLength(cnt, True), w / h, area / (w * h), area / hull]


def segment(gray, pts):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    masks = {"dark": (blur <= t).astype(np.uint8), "light": (blur > t).astype(np.uint8)}
    hits = {k: sum(int(m[min(max(int(round(y)), 0), h - 1), min(max(int(round(x)), 0), w - 1)])
                   for _, x, y in pts) for k, m in masks.items()}
    if hits["dark"] == hits["light"]:
        return None
    mask = masks["dark" if hits["dark"] > hits["light"] else "light"] * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in cnts if cv2.contourArea(c) >= MIN_AREA]


def orientation(cnt):
    m = cv2.moments(cnt)
    if m["m00"] <= 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    mu20, mu02, mu11 = (m[k] / m["m00"] for k in ("mu20", "mu02", "mu11"))
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    c = math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + c) / 2, (mu20 + mu02 - c) / 2
    if l2 <= 1e-9 or math.sqrt(l1 / l2) < MIN_ELONGATION:
        return None
    return cx, cy, theta


def wing_row(gray, cnt, head, tips, jp, i):
    """Return [patch_path, angle_right_rad, angle_left_rad] or None (reason counted)."""
    if len(tips) != 2:
        skipped["wing: need exactly 2 'mw' points in a single-male blob"] += 1
        return None
    m = cv2.moments(cnt)
    if m["m00"] <= 0:
        skipped["wing: degenerate blob"] += 1
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    ax, ay = head[1] - cx, head[2] - cy
    d = math.hypot(ax, ay)
    if d < 1e-6:
        skipped["wing: head coincides with centroid (body axis undefined)"] += 1
        return None
    ax, ay = ax / d, ay / d            # anterior unit vector (image coords, y down)
    rx, ry = -ay, ax                   # fly's right, dorsal view
    res = {}
    for _, tx, ty in tips:
        vx, vy = tx - cx, ty - cy
        n = math.hypot(vx, vy)
        side_val = vx * rx + vy * ry
        if n < 1e-6 or abs(side_val) < 1e-6 * n:
            skipped["wing: tip on body axis / at centroid (side ambiguous)"] += 1
            return None
        side = "right" if side_val > 0 else "left"
        if side in res:
            skipped["wing: both tips on the same side (left/right ambiguous)"] += 1
            return None
        cosang = max(-1.0, min(1.0, -(vx * ax + vy * ay) / n))  # vs posterior axis
        res[side] = math.acos(cosang)

    # head-up, centroid-centred 64x64 grayscale patch (real pixels only)
    alpha = math.degrees(math.atan2(-ay, ax))          # screen-CCW angle of anterior
    M = cv2.getRotationMatrix2D((cx, cy), 90.0 - alpha, 1.0)
    M[0, 2] += WING_PATCH / 2 - cx
    M[1, 2] += WING_PATCH / 2 - cy
    inv = cv2.invertAffineTransform(M)
    W = float(WING_PATCH)
    corners = np.array([[0, 0], [W, 0], [0, W], [W, W]], dtype=np.float64)
    src = corners @ inv[:, :2].T + inv[:, 2]
    h, w = gray.shape
    if (src < 0).any() or src[:, 0].max() > w - 1 or src[:, 1].max() > h - 1:
        skipped["wing: patch footprint crosses image border"] += 1
        return None
    patch = cv2.warpAffine(gray, M, (WING_PATCH, WING_PATCH), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = PATCH_DIR / "wing" / f"{jp.parent.name}__{jp.stem}__blob{i}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), patch):
        skipped["wing: patch write failed"] += 1
        return None
    return [out.as_posix(), res["right"], res["left"]]


def wing_report(stats):
    return (
        "wing_labels.csv: zero derivable rows, not written. Observed in single-male blobs: "
        f"{stats['male_blobs']} blobs, 'mw' points per blob = {dict(sorted(stats['mw_per_male'].items()))}. "
        "Rows require exactly two 'mw' points per male, on opposite sides of the body axis. "
        "If your wings are annotated differently (e.g. one 'mw' point per male, or a different "
        "label for wing tips), send the annotation convention."
    )


def inspect():
    labels, shapes = Counter(), Counter()
    jsons = sorted(IMAGES_DIR.rglob("*.json"))
    for p in jsons:
        for s in json.loads(p.read_text(encoding="utf-8")).get("shapes", []):
            labels[s.get("label")] += 1
            shapes[s.get("shape_type")] += 1
    print(f"{len(jsons)} JSON files; labels={dict(labels)}; shape_types={dict(shapes)}")


def main():
    if not IMAGES_DIR.is_dir():
        die(f"{IMAGES_DIR} not found. Run from the repo root.")
    if "--inspect" in sys.argv:
        return inspect()
    jsons = sorted(IMAGES_DIR.rglob("*.json"))
    if not jsons:
        die(f"no .json annotations under {IMAGES_DIR}")

    # ---- preflight: files whose annotations are outside the known vocabulary are skipped
    bad_files = {}
    for p in jsons:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            bad_files[p] = f"file skipped: unreadable JSON ({type(e).__name__})"
            continue
        shapes = d.get("shapes", [])
        unk = {s.get("label") for s in shapes} - KNOWN_LABELS
        if unk:
            bad_files[p] = f"file skipped: unknown labels {sorted(map(str, unk))}"
        elif any(s.get("shape_type") != "point" or len(s.get("points", [])) != 1 for s in shapes):
            bad_files[p] = "file skipped: non-point shapes present"
    for p, why in bad_files.items():
        print(f"WARNING: {p}: {why}", file=sys.stderr)

    rows = {k: [] for k in COLS}
    wing_stats = {"male_blobs": 0, "mw_per_male": Counter()}

    for jp in jsons:
        if jp in bad_files:
            skipped[bad_files[jp]] += 1
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        pts = [(s["label"], float(s["points"][0][0]), float(s["points"][0][1]))
               for s in data.get("shapes", []) if len(s.get("points", [])) == 1]
        if not pts:
            skipped["file skipped: no points"] += 1
            continue
        img = load_image(jp, data)
        if img is None:
            skipped["file skipped: image not found on disk via imagePath"] += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnts = segment(gray, pts)
        if cnts is None:
            skipped["file skipped: foreground polarity ambiguous"] += 1
            continue

        blob_pts = [[] for _ in cnts]
        outside = 0
        for lab, x, y in pts:
            dists = [cv2.pointPolygonTest(c, (x, y), True) for c in cnts]
            if dists and max(dists) >= 0:
                blob_pts[int(np.argmax(dists))].append((lab, x, y))
            else:
                outside += 1
        if outside:
            skipped["file skipped: annotated points outside all blobs"] += 1
            continue
        patches = {}

        for i, (cnt, bp) in enumerate(zip(cnts, blob_pts)):
            f = features(cnt)
            if f is None:
                skipped["blob skipped: degenerate geometry"] += 1
                continue
            heads = [p for p in bp if p[0] in HEAD_SEX]
            n = len(heads)
            if n in COUNT_NAME:
                rows["fly_count_labels.csv"].append(f + [COUNT_NAME[n]])
            else:
                skipped["fly_count: blob with 3+ heads has no valid label"] += 1
            if n != 1:
                continue
            sex = HEAD_SEX[heads[0][0]]
            other = "m" if sex == "female" else "f"
            if any(p[0].startswith(other) for p in bp):
                skipped["sex/orientation: blob has points of both sexes"] += 1
                continue
            rows["sex_labels.csv"].append(f + [sex])
            if sex == "male":
                tips = [p for p in bp if p[0] == "mw"]
                wing_stats["male_blobs"] += 1
                wing_stats["mw_per_male"][len(tips)] += 1
                wr = wing_row(gray, cnt, heads[0], tips, jp, i)
                if wr:
                    rows["wing_labels.csv"].append(wr)

            mo = orientation(cnt)
            if mo is None:
                skipped["orientation: blob not elongated / degenerate"] += 1
                continue
            cx, cy, theta = mo
            hx, hy = heads[0][1] - cx, heads[0][2] - cy
            proj = hx * math.cos(theta) + hy * math.sin(theta)
            if abs(proj) < 1e-6 * max(1.0, math.hypot(hx, hy)):
                skipped["orientation: flip ambiguous (head on axis centre)"] += 1
                continue
            half = PATCH_SIZE // 2
            x0, y0 = int(round(cx)) - half, int(round(cy)) - half
            H, W = img.shape[:2]
            if x0 < 0 or y0 < 0 or x0 + PATCH_SIZE > W or y0 + PATCH_SIZE > H:
                skipped["orientation: patch crosses image border"] += 1
                continue
            out = PATCH_DIR / f"{jp.parent.name}__{jp.stem}__blob{i}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), img[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]):
                skipped["orientation: patch write failed"] += 1
                continue
            rows["orientation_labels.csv"].append(
                [out.as_posix(), theta, 1 if proj < 0 else 0])

    # ---- write only CSVs that have real rows (never header-only)
    failed = []
    print("\n===== SUMMARY =====")
    print(f"JSON files found: {len(jsons)}")
    for name in COLS:
        if rows[name]:
            write_csv(name, rows[name])
            print(f"  {name}: {len(rows[name])} rows -> {IMAGES_DIR / name}")
        elif name == "wing_labels.csv":
            failed.append(wing_report(wing_stats))
            if (IMAGES_DIR / name).exists():
                failed.append("an existing wing_labels.csv was left untouched and may be stale.")
        else:
            failed.append(f"{name}: zero derivable rows, not written")
    for k, v in skipped.most_common():
        if v:
            print(f"  skipped {v:6d}  {k}")
    if rows["fly_count_labels.csv"]:
        print("  fly_count label distribution:",
              dict(Counter(r[-1] for r in rows["fly_count_labels.csv"])))
    if rows["sex_labels.csv"]:
        print("  sex label distribution:",
              dict(Counter(r[-1] for r in rows["sex_labels.csv"])))

    if failed:
        print("\nUNRESOLVED:", file=sys.stderr)
        for m in failed:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()