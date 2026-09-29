# PROJECT_AUDIT.md
## Real-time Detailed Video Analysis of Fruit Flies — Group 37
**Audit date:** September 29, 2026  
**Auditor:** Automated + manual inspection via `audit_inspect.py`

---

## A. Existing Project Structure

```
ML_Mini_Project_37/
├── preprocessing.py       # Video I/O, thresholding, contour extraction
├── fly_count.py           # FlyCount Decision Tree classifier
├── sex_classification.py  # Sex classification (Logistic Regression)
├── orientation.py         # Body orientation (moments + HOG disambiguation)
├── wing_angle.py          # Wing-angle regression (HOG + PCA + LR)
├── main.py                # Integrated pipeline runner
├── validate_dataset.py    # Dataset validation script (added in audit)
├── models/
│   ├── fly_count_model.joblib
│   ├── wing_angle_right_model.joblib
│   └── wing_angle_left_model.joblib
├── output/                # Generated visualizations
├── tests/                 # Automated pytest suite (added in audit)
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_fly_count.py
│   ├── test_orientation.py
│   ├── test_wing_angle.py
│   └── test_integration.py
├── requirements.txt
├── README.md
└── .gitignore
```

**Missing (real data not available):**
- `input/` directory — CS229 labeled dataset (~2 GB, Dropbox)
- `models/sex_clf_model.joblib` — needs sex_labels.csv
- `models/orientation_model.joblib` — needs orientation_labels.csv

---

## B. Purpose of Every Python File

| File | Purpose |
|------|---------|
| `preprocessing.py` | Video loading, grayscale conversion, Otsu thresholding, contour extraction, visualization |
| `fly_count.py` | Decision Tree to classify contours as zero/one/two flies; rule-based fallback |
| `sex_classification.py` | StandardScaler + Logistic Regression for male/female classification |
| `orientation.py` | Image-moments body-axis angle; HOG+PCA+LogReg to resolve 180° ambiguity |
| `wing_angle.py` | Gaussian blur + CLAHE + HOG + PCA + Linear Regression for wing angles |
| `main.py` | Integrated pipeline: loads models, processes video frames, annotates, measures FPS |
| `validate_dataset.py` | Validates CSV annotations and video files before training |
| `audit_inspect.py` | One-time audit diagnostics script |
| `audit_blob_debug.py` | Bug investigation script (can be deleted after audit) |

---

## C. Actual Implementation of Every Module

### preprocessing.py
- `load_video(path)` → `cv2.VideoCapture` + properties dict
- `read_frames(cap, max_frames)` → generator yielding uint8 grayscale frames
- `compute_background(cap, n_samples)` → mean of first N frames as background model
- `threshold_frame(frame, background, blur_ksize)` → binary mask via Otsu
- `extract_contours(mask, min_area, max_area)` → filtered external contours
- `contour_features(contour)` → dict: area, perimeter, aspect_ratio, extent, solidity
- `make_synthetic_video(path, n_frames, width, height)` → demo video

### fly_count.py
- Features: area, perimeter, aspect_ratio, extent, solidity
- `DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced")`
- `LabelEncoder` fitted on `["zero","one","two"]` → alphabetical: one=0, two=1, zero=2
- Fallback: `rule_based_predict()` using area < 80 → zero, solidity > 0.85 → one, else → two
- Saves `{"clf": ..., "le": ...}` bundle

### sex_classification.py
- Features: area, perimeter, aspect_ratio, extent, solidity
- `Pipeline(StandardScaler, LogisticRegression(L2, lbfgs, balanced))`
- `classify_pair(c_a, c_b, model)` returns consistent male/female assignment
- Strict data check — no fabrication

### orientation.py
- Stage 1: `moments_orientation(patch)` → `0.5 * arctan2(2*mu11, mu20-mu02)` range `(-pi/2, pi/2)`
- Stage 2: `Pipeline(PCA(30), LogisticRegression)` predicts flip ∈ {0,1}
- `predict_orientation(patch, model)` → Stage 1 alone or Stage 1+2

### wing_angle.py
- Preprocessing: Gaussian blur (3×3) + CLAHE
- Wing split: left = columns 0..27, right = columns 36..63 (WING_FRACTION=0.45)
- HOG: 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys norm
- `Pipeline(PCA(30), LinearRegression)` — separate models for left and right
- Data check strict — no regression without labels

### main.py
- `load_all_models()` → status dict with `ok` flags
- `process_frame(frame, background, models)` → per-contour predictions
- `annotate_frame(frame, result, models, fps)` → BGR annotated frame with HUD
- FPS measured per frame via `time.perf_counter()`
- Per-module N/A overlay when model unavailable

---

## D. Algorithms Used

| Module | Algorithm | Justification |
|--------|-----------|---------------|
| Thresholding | Otsu's method | Automatically optimal for bimodal histograms |
| FlyCount | Decision Tree (depth 5) | Interpretable; works on small geometric features |
| Sex | Logistic Regression + StandardScaler | Probabilistic; needed for classify_pair() |
| Orientation Stage 1 | Image moments | Closed-form, no data needed |
| Orientation Stage 2 | HOG + PCA + LogReg | Appearance-based disambiguation |
| Wing angle | HOG + PCA + Linear Regression | CS229 paper approach |

---

## E. Expected Input / Output of Every Module

| Module | Input | Output |
|--------|-------|--------|
| `preprocessing.py` | .mp4 path | contours, masks, features |
| `fly_count.py` | contour | "zero"/"one"/"two" label |
| `sex_classification.py` | two contours | male/female assignment dict |
| `orientation.py` | 64×64 patch | angle in radians |
| `wing_angle.py` | 64×64 patch | (angle_right, angle_left) in radians |
| `main.py` | video path | annotated frames, FPS, summary PNG |

---

## F. Missing Files, Datasets, Annotations, Dependencies, Models

| Item | Status | Impact |
|------|--------|--------|
| `input/` directory | MISSING | Real training impossible |
| `input/images/fly_count_labels.csv` | MISSING | DT trains on synthetic only |
| `input/images/sex_labels.csv` | MISSING | Sex module unavailable |
| `input/images/orientation_labels.csv` | MISSING | Stage 2 disambiguation unavailable |
| `input/images/wing_labels.csv` | MISSING | Wing regression unavailable |
| `input/video/test*.mp4` | MISSING | Demo uses synthetic video |
| `models/sex_clf_model.joblib` | MISSING | Sex predictions show N/A |
| `models/orientation_model.joblib` | MISSING | Moments baseline used only |

---

## G. Bugs Found and Fixed

### BUG-01 — CRITICAL — `preprocessing.py::threshold_frame`
**Location:** `threshold_frame()`, the `background=None` branch  
**Problem:** When no background is provided, Otsu was applied to the raw frame with `THRESH_BINARY`. Flies are dark on a light background, so the background (large area, high pixel value) was treated as foreground. This created a single huge contour (~76,000 px²) exceeding `max_area=5000`, so `extract_contours()` returned an empty list.  
**Evidence:** `audit_blob_debug.py` showed 0 contours from a frame with a clear single fly.  
**Fix:** Use `THRESH_BINARY_INV` when `background=None` so dark fly pixels become 255 (foreground).  
**Verified:** After fix, single blob → 1 contour of area 264 px². ✓

### BUG-02 — Previously fixed — `fly_count.py::make_synthetic_dataset`
**Problem:** `y` array was constructed as `[0,1,2] * n_per_class` with the assumption 0=zero, 1=one, 2=two, but `LabelEncoder` sorts alphabetically giving one=0, two=1, zero=2. This caused label inversion.  
**Fix (already applied):** Use `le.transform(labels_str)` instead of hardcoded integers.  
**Verified:** `audit_inspect.py` confirms one=0, two=1, zero=2. ✓

---

## H. Data Leakage Risks

| Risk | Status |
|------|--------|
| PCA fitted before train/test split | **NOT present** — PCA is inside `sklearn.Pipeline`, fitted only on `X_train` |
| StandardScaler fitted on full dataset | **NOT present** — inside Pipeline, fitted on `X_train` only |
| Test data used for hyperparameter tuning | **NOT present** — depth=5 is fixed |
| Synthetic data mixed with real results | **NOT present** — synthetic data results are clearly labelled |
| Frame-level split on same-video data | **Risk exists** — if real video data is used, frames from same video may appear in both splits. Mitigated by: dataset is small enough that grouping is not critical; `GroupShuffleSplit` recommended when real data is available |

---

## I. Model Training and Evaluation Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| FlyCount trained on synthetic data only | HIGH | 99.12% accuracy is on synthetic, NOT real video |
| Wing angle trained on synthetic data only | HIGH | MAE ~25° is on synthetic, NOT real |
| PCA retains only 46% variance with 30 components | MEDIUM | Expected for HOG; can increase n_components when more data available |
| No cross-validation for FlyCount or wing regression | LOW | Added CV to sex classifier; DT uses fixed train/test split |
| No held-out test set for real data | BLOCKED | Blocked by missing dataset |

---

## J. Integration Problems

| Issue | Status |
|-------|--------|
| `predict_wing_angles` returns `None` when models absent — handled in main.py | FIXED |
| `predict_sex` signature changed (now returns tuple) — main.py updated | FIXED |
| `orient_model` not passed through to `process_frame` | FIXED |
| `load_wing_models` returns `(None, None)` gracefully | FIXED |
| All modules independently runnable | ✓ |

---

## K. Documentation and Submission Issues

| Issue | Status |
|-------|--------|
| README missing exact install and run commands | FIXED in updated README.md |
| No automated tests | FIXED — tests/ directory added |
| No dataset validation script | FIXED — validate_dataset.py added |
| Audit document missing | FIXED — this file |
| Submission checklist missing | FIXED — SUBMISSION_CHECKLIST.md added |
| Temporary audit scripts in root | LOW — audit_inspect.py, audit_blob_debug.py |

---

## L. Prioritized Fix List

### Critical (prevents correct execution)
| # | File | Issue | Fix | Tested |
|---|------|-------|-----|--------|
| C1 | `preprocessing.py` | `threshold_frame` wrong polarity without background | Use `THRESH_BINARY_INV` when `background=None` | ✓ |

### High (major correctness problem)
| # | File | Issue | Fix | Tested |
|---|------|-------|-----|--------|
| H1 | `fly_count.py` | LabelEncoder index mismatch (previously fixed) | `le.transform(labels_str)` | ✓ |
| H2 | All modules | No automated tests | `tests/` directory added | ✓ |
| H3 | All modules | No dataset validation | `validate_dataset.py` added | ✓ |

### Medium (reliability / maintainability)
| # | File | Issue | Fix | Tested |
|---|------|-------|-----|--------|
| M1 | `wing_angle.py` | PCA only 46% variance with 30 components | Documented; increase n_components with more data | Documented |
| M2 | `main.py` | No `--max_area` / `--min_area` CLI options | Low priority for demo | N/A |

### Low (documentation / cosmetic)
| # | File | Issue | Fix | Tested |
|---|------|-------|-----|--------|
| L1 | root | Temporary audit scripts in root dir | Can delete after submission | N/A |
| L2 | `requirements.txt` | Versions pinned tightly | Acceptable for reproducibility | ✓ |
| L3 | `.gitignore` | `synthetic_test.mp4` not excluded | Add `synthetic_test.mp4` | ✓ |
