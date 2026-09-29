# Real-time Detailed Video Analysis of Fruit Flies

**ML Mini Project — Group 37**

A modular Python pipeline for analyzing grayscale videos of fruit flies. The system estimates fly count, male/female identity, body orientation, and male wing angles using classical computer vision and machine learning.

---

## Team Members & Responsibilities

| Member | Modules |
|--------|---------|
| Member 1 | `fly_count.py` — FlyCount classifier, `wing_angle.py` — Wing Angle regression |
| Member 2 | `sex_classification.py` — Male/female identity, `orientation.py` — Body orientation estimation |

---

## Project Structure

```
ML_Mini_Project_37/
├── preprocessing.py      # Video loading, grayscale, thresholding, contour extraction
├── fly_count.py          # Decision tree classifier: 0, 1, or 2 flies per contour
├── wing_angle.py         # HOG + PCA + Linear Regression for wing angle prediction
├── sex_classification.py # Male/female identity classification
├── orientation.py        # Body orientation estimation
├── main.py               # Full pipeline: ties all modules together
├── models/               # Saved trained models (.joblib files)
├── output/               # Output plots, evaluation results
├── input/                # Place video and labeled data here (see Data Setup)
│   ├── video/            # .mp4 video files (test1.mp4 … test5.mp4)
│   └── images/           # Labeled image patches for training
├── requirements.txt
└── README.md
```

---

## Data Setup (Required Before Training)

The labeled training data (~2 GB) is **not included** in this repo. Download it from the original CS229 project:

> **Dropbox link:** https://www.dropbox.com/sh/78inyvw2ouut74a/AACc1DYrC1G0UxujwT-6ryRKa?dl=0

1. Download and unzip the archive.
2. Place the resulting `input/` folder at the root of this project:
   - `input/video/test1.mp4` … `test5.mp4`
   - `input/images/` — labeled patches for fly count, sex, orientation, and wing angle

Without this data, all modules fall back to **synthetic data** automatically so the code can still be demonstrated end-to-end.

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.8+ recommended.

---

## Usage

### Run the full pipeline (demo mode — works without data)
```bash
python main.py
```

### Run with a real video
```bash
python main.py --video input/video/test4.mp4
```

### Train and evaluate fly count classifier only
```bash
python fly_count.py
```

### Train and evaluate wing angle regressor only
```bash
python wing_angle.py
```

### Run preprocessing and visualize contours
```bash
python preprocessing.py --video input/video/test4.mp4
```

---

## Module Descriptions

### `preprocessing.py`
- Loads video frames using OpenCV
- Converts to grayscale
- Applies Otsu's thresholding to isolate flies from background
- Extracts contours using `cv2.findContours`
- Visualizes: raw frame, thresholded mask, and contours overlaid

### `fly_count.py` — Member 1
- Computes geometric features from each contour: area, perimeter, aspect ratio, extent, solidity
- Labels contours as `"zero"`, `"one"`, or `"two"` flies
- Trains a **Decision Tree classifier** (scikit-learn)
- Evaluates with accuracy, classification report, and confusion matrix
- Saves trained model to `models/fly_count_model.joblib`
- Exports a decision tree diagram to `output/tree_fly_count.png`

### `wing_angle.py` — Member 1
- Extracts the male fly's region of interest (ROI) from a frame
- Crops left and right wing sub-regions
- Extracts **HOG (Histogram of Oriented Gradients)** features via scikit-image
- Applies **PCA** to reduce HOG feature dimensionality
- Trains a **Linear Regression** model to predict left/right wing angles (radians)
- Evaluates using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
- Saves model and PCA transform to `models/`

### `sex_classification.py` — Member 2
- Classifies each detected fly as male or female based on contour shape and intensity features

### `orientation.py` — Member 2
- Estimates the body orientation angle of each fly using image moments

### `main.py`
- Orchestrates the full pipeline frame-by-frame
- Loads video → preprocessing → fly count → sex classification → orientation → wing angle
- Saves annotated output frames and prints per-frame results

---

## Algorithms Used

| Task | Algorithm | Why |
|------|-----------|-----|
| Fly count | Decision Tree | Interpretable; works well with small geometric feature sets |
| Wing angle | Linear Regression + HOG + PCA | HOG captures local gradient structure in wing regions; PCA prevents overfitting |
| Thresholding | Otsu's method | Automatically finds optimal threshold for bimodal grayscale histograms |

---

## Member 1 Work

**Stage 1 — FlyCount Classifier (`fly_count.py`)**
Contour-based feature extraction (area, perimeter, aspect ratio, extent, solidity) and a Decision Tree to classify contour blobs as containing zero, one, or two flies. Handles the case where two touching flies merge into a single large contour.

**Stage 4 — Wing Angle Regression (`wing_angle.py`)**
Male fly ROI extraction, wing sub-region cropping, HOG feature computation, PCA dimensionality reduction, and Linear Regression to predict left and right wing angles. Evaluation uses MAE and RMSE on a held-out test split.

---

## Member 2 Work

**Sex Classification (`sex_classification.py`)**
Classifies each detected fly contour as male or female using contour area, aspect ratio, and intensity statistics.

**Orientation Estimation (`orientation.py`)**
Estimates fly body orientation angle using image moments on the binary fly patch.

---

## References

- Original CS229 project: https://github.com/sgherbst/cs229-project
- Dalal & Triggs (2005) — HOG features for human detection
- scikit-learn Decision Tree documentation: https://scikit-learn.org/stable/modules/tree.html
