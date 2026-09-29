# Real-time Detailed Video Analysis of Fruit Flies

**ML Mini Project — Group 37 | UE24CS352A Machine Learning**

A modular Python pipeline that analyses grayscale video of two fruit flies
and estimates fly count, body orientation, sex, and male wing angles using
classical computer vision and machine learning.

---

## Team Members & Responsibilities

| Member | Modules |
|--------|---------|
| Member 1 | `fly_count.py` — FlyCount Decision Tree classifier |
|           | `wing_angle.py` — HOG + PCA + Linear Regression wing-angle regression |
| Member 2 | `sex_classification.py` — Logistic Regression sex classifier |
|           | `orientation.py` — Image-moments baseline + HOG disambiguation |

---

## What Each Module Does

| Module | Algorithm | Status without labeled data |
|--------|-----------|----------------------------|
| `fly_count.py` | Decision Tree on 5 contour geometry features | Rule-based fallback (area + solidity thresholds) |
| `orientation.py` | Stage 1: image moments. Stage 2: HOG + PCA + Logistic Regression | Stage 1 always runs; Stage 2 needs `orientation_labels.csv` |
| `sex_classification.py` | StandardScaler + Logistic Regression | Disabled — no fabrication |
| `wing_angle.py` | HOG + PCA + Linear Regression | Preprocessing visualization runs; regression disabled |

---

## Project Structure

```
ML_Mini_Project_37/
├── main.py               # Full integrated pipeline (run this)
├── preprocessing.py      # Video I/O, grayscale, Otsu threshold, contours
├── fly_count.py          # FlyCount: Decision Tree classifier
├── orientation.py        # Orientation: moments baseline + HOG disambiguation
├── sex_classification.py # Sex: Logistic Regression (needs labeled data)
├── wing_angle.py         # Wing angle: HOG + PCA + Linear Regression
├── models/               # Saved .joblib model files (auto-created)
├── output/               # Annotated frames + evaluation plots (auto-created)
├── input/                # Place downloaded data here (see Data Setup)
│   ├── video/            # test1.mp4 … test5.mp4
│   └── images/           # Labeled patches and CSV files
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Python version
Python **3.8 or higher** is required. Check yours:
```bash
python --version
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

All packages with pinned versions:
```
numpy==1.24.4
opencv-python==4.8.1.78
scikit-learn==1.3.2
matplotlib==3.7.5
scikit-image==0.21.0
joblib==1.3.2
tqdm==4.66.1
imutils==0.5.4
```

> If you have a newer environment and want flexible versions, you can also run:
> `pip install numpy opencv-python scikit-learn matplotlib scikit-image joblib imutils`

---

## Data Setup (Required for full supervised training)

The labeled training data (~2 GB) is **not included** in this repo.
Download it from the original CS229 project:

> **Dropbox:** https://www.dropbox.com/sh/78inyvw2ouut74a/AACc1DYrC1G0UxujwT-6ryRKa?dl=0

1. Download and unzip.
2. Place the `input/` folder at the root of this project:
   ```
   ML_Mini_Project_37/input/video/test1.mp4
   ML_Mini_Project_37/input/video/test4.mp4   ← default demo video
   ML_Mini_Project_37/input/images/           ← labeled patches
   ```

**Without this data the project still runs** — FlyCount uses a rule-based
fallback, orientation uses the moments baseline, and sex/wing-angle modules
print a clear "unavailable" message instead of fabricating predictions.

---

## Running the Project

### Quickstart — synthetic demo (no data needed)
```bash
python main.py
```
Generates a synthetic video of two moving ellipses, processes 10 frames,
prints per-frame results, and saves annotated PNGs to `output/`.

### Process a real video
```bash
python main.py --video input/video/test4.mp4
```

### Process more frames
```bash
python main.py --video input/video/test4.mp4 --frames 30
```

### Live OpenCV preview window (needs a desktop / display server)
```bash
python main.py --video input/video/test4.mp4 --display
```
Press **q** in the window to quit early.

### Save outputs to a custom directory
```bash
python main.py --video input/video/test4.mp4 --save_dir results/
```

### Retrain the FlyCount model then run
```bash
python main.py --train
```

---

## Running Individual Modules

### FlyCount classifier (Member 1 — Stage 1)
```bash
# No data: runs rule-based baseline + Decision Tree demo on synthetic data
python fly_count.py

# With real labeled data:
python fly_count.py --data input/images/fly_count_labels.csv
```
Expected CSV columns: `area, perimeter, aspect_ratio, extent, solidity, label`
Labels: `zero` | `one` | `two`

### Wing-angle regression (Member 1 — Stage 4)
```bash
# No data: runs preprocessing + visualization only (no regression)
python wing_angle.py

# With real labeled data:
python wing_angle.py --data input/images/wing_labels.csv
```
Expected CSV columns: `patch_path, angle_right_rad, angle_left_rad`

### Orientation estimation (Member 2)
```bash
# No data: moments baseline runs + limitations explained
python orientation.py

# With real labeled data (for 180° disambiguation):
python orientation.py --data input/images/orientation_labels.csv
```
Expected CSV columns: `patch_path, moment_angle_rad, flip`
`flip` is `0` (angle correct) or `1` (add pi to fix direction).

### Sex classification (Member 2)
```bash
# No data: strict check, no fabrication, classify_pair() demo shown
python sex_classification.py

# With real labeled data:
python sex_classification.py --data input/images/sex_labels.csv
```
Expected CSV columns: `area, perimeter, aspect_ratio, extent, solidity, label`
Labels: `male` | `female`

### Preprocessing standalone
```bash
python preprocessing.py                        # synthetic video
python preprocessing.py --video input/video/test4.mp4 --frames 5
```

---

## What the Pipeline Outputs

After running `main.py`:

| File | Description |
|------|-------------|
| `output/frame_0000.png` … | Annotated frames with contours, count, orientation arrow, sex label, wing-angle arrows |
| `output/pipeline_summary.png` | Grid of all processed frames in one figure |
| `output/confusion_matrix_fly_count.png` | FlyCount evaluation (when trained) |
| `output/tree_fly_count.png` | Decision tree diagram |
| `output/wing_angle_right_scatter.png` | True vs predicted scatter (when trained) |
| `output/orientation_baseline_demo.png` | Moments baseline visualization |
| `output/wing_angle_preprocessing_demo.png` | HOG pipeline visualization |

Console output includes per-frame:
- Fly count, contour labels
- Orientation angle (degrees)
- Sex label (or `N/A` with reason)
- Wing angles right/left (or `N/A` with reason)
- Per-frame FPS and mean FPS at the end

---

## HUD Legend (annotated frames)

| Color | Meaning |
|-------|---------|
| Green outline | Single fly contour |
| Orange outline | Two merged flies |
| Grey outline | Noise (zero flies) |
| Cyan arrow | Body orientation axis |
| Orange arrow | Right wing angle |
| Blue arrow | Left wing angle |
| `M` / `F` label | Sex prediction (only when model available) |
| `sex:N/A` | Sex module unavailable — no trained model |
| `wing:N/A` | Wing module unavailable — no trained model |

---

## Honest Module Status

| Module | Supervised model trains? | What runs without data |
|--------|--------------------------|------------------------|
| FlyCount | Yes — on `fly_count_labels.csv` | Rule-based area+solidity fallback |
| Orientation Stage 1 | N/A (geometry, no training) | Fully functional — moments always run |
| Orientation Stage 2 | Yes — on `orientation_labels.csv` | Skipped; 180° ambiguity note shown |
| Sex classifier | Yes — on `sex_labels.csv` | Disabled, `N/A` shown on frame |
| Wing-angle regression | Yes — on `wing_labels.csv` | Preprocessing + HOG vis only |

No module fabricates predictions or reports accuracy numbers on synthetic data
as if they were real results.

---

## References

- Original CS229 project: https://github.com/sgherbst/cs229-project
- Dalal & Triggs (2005) — Histograms of Oriented Gradients for Human Detection
- scikit-learn documentation: https://scikit-learn.org/stable/
