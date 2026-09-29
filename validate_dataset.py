"""
validate_dataset.py
-------------------
Dataset validation script for the Fruit Fly Analysis project.

Run this script after placing the labeled data in input/ to verify:
  - All expected CSV files are present and readable
  - Label distributions are reported
  - Image patches referenced in CSVs are loadable
  - Annotation values are within valid ranges
  - Duplicate rows are flagged
  - Class imbalance is reported

Usage
-----
    python validate_dataset.py
    python validate_dataset.py --input_dir input/images

No labels are fabricated.  If a file is missing the script reports it
clearly and continues checking what is available.
"""

import os
import sys
import csv
import argparse
from collections import Counter

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str):
    print(f"\n{'─'*58}")
    print(f"  {title}")
    print(f"{'─'*58}")


def _check_csv(csv_path: str, required_columns: set, label_col: str = None,
               valid_labels: set = None, numeric_cols: dict = None,
               image_col: str = None) -> dict:
    """
    Generic CSV validation.

    Parameters
    ----------
    csv_path        : path to the CSV file
    required_columns: set of column names that must be present
    label_col       : name of the label column (optional)
    valid_labels    : set of acceptable label strings (optional)
    numeric_cols    : dict {col_name: (min_val, max_val)} for range checks
    image_col       : column name containing image paths (optional)

    Returns
    -------
    report dict with counts and issues
    """
    report = {
        "file":          csv_path,
        "exists":        False,
        "readable":      False,
        "n_rows":        0,
        "n_valid":       0,
        "n_invalid":     0,
        "n_duplicates":  0,
        "label_counts":  {},
        "missing_images":0,
        "range_errors":  0,
        "issues":        [],
    }

    if not os.path.isfile(csv_path):
        report["issues"].append(f"FILE NOT FOUND: {csv_path}")
        print(f"  [MISS] {csv_path}")
        return report

    report["exists"] = True

    try:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        report["readable"] = True
    except Exception as e:
        report["issues"].append(f"READ ERROR: {e}")
        print(f"  [ERR]  Cannot read {csv_path}: {e}")
        return report

    if not rows:
        report["issues"].append("CSV is empty (no data rows).")
        print(f"  [WARN] {csv_path} is empty.")
        return report

    report["n_rows"] = len(rows)

    # Column check
    actual_cols = set(rows[0].keys())
    missing_cols = required_columns - actual_cols
    if missing_cols:
        report["issues"].append(f"Missing columns: {missing_cols}")
        print(f"  [ERR]  Missing columns: {missing_cols}")

    # Duplicate rows
    row_strings = [str(sorted(r.items())) for r in rows]
    report["n_duplicates"] = len(row_strings) - len(set(row_strings))
    if report["n_duplicates"] > 0:
        report["issues"].append(f"{report['n_duplicates']} duplicate rows found.")

    # Label distribution
    if label_col and label_col in actual_cols:
        labels = [r[label_col].strip().lower() for r in rows]
        report["label_counts"] = dict(Counter(labels))
        if valid_labels:
            bad_labels = {l for l in labels if l not in valid_labels}
            if bad_labels:
                report["issues"].append(f"Invalid labels: {bad_labels}")

    # Numeric range checks
    if numeric_cols:
        for col, (lo, hi) in numeric_cols.items():
            if col not in actual_cols:
                continue
            for i, row in enumerate(rows):
                try:
                    val = float(row[col])
                    if not (lo <= val <= hi):
                        report["range_errors"] += 1
                        if report["range_errors"] <= 3:
                            report["issues"].append(
                                f"Row {i}: {col}={val:.4f} out of range [{lo}, {hi}]")
                except ValueError:
                    report["range_errors"] += 1
                    report["issues"].append(f"Row {i}: {col} is not numeric: '{row[col]}'")

    # Image patch existence
    if image_col and image_col in actual_cols:
        missing = 0
        for row in rows:
            p = row[image_col].strip()
            full = p if os.path.isabs(p) else os.path.join(ROOT, p)
            if not os.path.isfile(full):
                missing += 1
        report["missing_images"] = missing
        if missing > 0:
            report["issues"].append(
                f"{missing}/{len(rows)} image patches not found.")

    report["n_valid"]   = report["n_rows"] - report["n_invalid"]
    return report


def _print_report(report: dict):
    status = "[OK]" if not report["issues"] else "[ISSUES]"
    print(f"\n  {status} {os.path.basename(report['file'])}")
    print(f"    Rows      : {report['n_rows']}")
    if report["label_counts"]:
        print(f"    Labels    : {report['label_counts']}")
        # Class imbalance check
        counts = list(report["label_counts"].values())
        if len(counts) > 1:
            ratio = max(counts) / max(min(counts), 1)
            if ratio > 3:
                print(f"    [WARN] Class imbalance ratio {ratio:.1f}:1 — consider balancing.")
    if report["n_duplicates"] > 0:
        print(f"    Duplicates: {report['n_duplicates']}")
    if report["range_errors"] > 0:
        print(f"    Range errs: {report['range_errors']}")
    if report["missing_images"] > 0:
        print(f"    Miss imgs : {report['missing_images']}")
    for iss in report["issues"]:
        print(f"    [!] {iss}")


# ---------------------------------------------------------------------------
# Per-module validators
# ---------------------------------------------------------------------------

def validate_fly_count(images_dir: str):
    _sep("FlyCount Labels  (fly_count_labels.csv)")
    path = os.path.join(images_dir, "fly_count_labels.csv")
    report = _check_csv(
        path,
        required_columns={"area", "perimeter", "aspect_ratio", "extent", "solidity", "label"},
        label_col="label",
        valid_labels={"zero", "one", "two"},
        numeric_cols={
            "area":         (0, 10000),
            "perimeter":    (0, 1000),
            "aspect_ratio": (0, 20),
            "extent":       (0, 1),
            "solidity":     (0, 1),
        },
    )
    _print_report(report)
    return report


def validate_sex(images_dir: str):
    _sep("Sex Labels  (sex_labels.csv)")
    path = os.path.join(images_dir, "sex_labels.csv")
    report = _check_csv(
        path,
        required_columns={"area", "perimeter", "aspect_ratio", "extent", "solidity", "label"},
        label_col="label",
        valid_labels={"male", "female"},
        numeric_cols={
            "area":         (0, 10000),
            "aspect_ratio": (0, 20),
            "solidity":     (0, 1),
        },
    )
    _print_report(report)
    return report


def validate_orientation(images_dir: str):
    _sep("Orientation Labels  (orientation_labels.csv)")
    path = os.path.join(images_dir, "orientation_labels.csv")
    report = _check_csv(
        path,
        required_columns={"patch_path", "moment_angle_rad", "flip"},
        label_col="flip",
        valid_labels={"0", "1"},
        numeric_cols={
            "moment_angle_rad": (-1.6, 1.6),   # (-pi/2, pi/2)
            "flip":             (0, 1),
        },
        image_col="patch_path",
    )
    _print_report(report)
    return report


def validate_wing_angle(images_dir: str):
    _sep("Wing-Angle Labels  (wing_labels.csv)")
    path = os.path.join(images_dir, "wing_labels.csv")
    report = _check_csv(
        path,
        required_columns={"patch_path", "angle_right_rad", "angle_left_rad"},
        numeric_cols={
            "angle_right_rad": (0, 3.15),   # (0, pi)
            "angle_left_rad":  (0, 3.15),
        },
        image_col="patch_path",
    )
    _print_report(report)
    return report


def validate_videos(video_dir: str):
    _sep("Video Files  (input/video/)")
    if not os.path.isdir(video_dir):
        print(f"  [MISS] {video_dir} does not exist.")
        print("         Download from: https://www.dropbox.com/sh/78inyvw2ouut74a/...")
        return

    video_files = [f for f in os.listdir(video_dir)
                   if f.lower().endswith((".mp4", ".avi", ".mov"))]
    print(f"  Found {len(video_files)} video file(s): {video_files}")

    for vf in video_files:
        vpath = os.path.join(video_dir, vf)
        cap = cv2.VideoCapture(vpath)
        if cap.isOpened():
            fc   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps  = cap.get(cv2.CAP_PROP_FPS)
            w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [OK]  {vf}: {w}×{h} @ {fps:.1f}fps, {fc} frames")
            cap.release()
        else:
            print(f"  [ERR] Cannot open {vf}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir: str = "input"):
    images_dir = os.path.join(input_dir, "images")
    video_dir  = os.path.join(input_dir, "video")

    print("=" * 58)
    print("  Dataset Validation — Fruit Fly Analysis Project")
    print("=" * 58)

    if not os.path.isdir(input_dir):
        print(f"\n  [MISS] input/ directory not found at: {input_dir}")
        print("         No real data is available.")
        print("         Download the dataset:")
        print("         https://www.dropbox.com/sh/78inyvw2ouut74a/AACc1DYrC1G0UxujwT-6ryRKa?dl=0")
        print("\n  All modules fall back to synthetic/rule-based mode without this data.")
        print("  The pipeline itself is functional — run:  python main.py")
        return

    validate_fly_count(images_dir)
    validate_sex(images_dir)
    validate_orientation(images_dir)
    validate_wing_angle(images_dir)
    validate_videos(video_dir)

    print("\n" + "=" * 58)
    print("  Validation complete.")
    print("  Fix any [!] issues before training supervised models.")
    print("=" * 58)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate dataset CSVs and video files for the fruit fly project")
    parser.add_argument("--input_dir", type=str, default="input",
                        help="Path to the input/ directory (default: input/)")
    args = parser.parse_args()
    main(args.input_dir)
