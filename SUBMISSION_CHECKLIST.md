# SUBMISSION_CHECKLIST.md
## Real-time Detailed Video Analysis of Fruit Flies — Group 37
**Course:** UE24CS352A Machine Learning  
**Last updated:** September 29, 2026

---

## A. GitHub Repository Checklist

- [x] Repository is public: https://github.com/thrupthi912/ML_Mini_Project_37
- [x] All source files committed: preprocessing.py, fly_count.py, sex_classification.py, orientation.py, wing_angle.py, main.py
- [x] requirements.txt present with pinned versions
- [x] README.md present with installation and run instructions
- [x] .gitignore excludes: *.mp4, models/, output/, input/, __pycache__/
- [x] No credentials, API keys, or private data committed
- [x] No large binary datasets committed
- [ ] Final commit pushed before submission deadline

---

## B. Code and Execution Checklist

- [x] All modules independently runnable (`python fly_count.py`, etc.)
- [x] `python main.py` runs to completion without errors (verified on synthetic video)
- [x] FPS measured and reported (mean FPS: ~44–75 on Windows, 320×240 synthetic video)
- [x] All modules handle missing models gracefully (N/A overlay, no crash)
- [x] All modules handle missing data gracefully (clear messages, no fabrication)
- [x] Error handling wraps every per-contour prediction in main.py
- [x] `cap.release()` called after all video operations
- [x] Output saved to `output/` directory
- [x] `--display` flag for live OpenCV window (requires desktop)

---

## C. Dataset and Annotation Checklist

- [ ] CS229 dataset downloaded from Dropbox (2 GB)
- [ ] `input/video/` contains test1.mp4–test5.mp4
- [ ] `input/images/fly_count_labels.csv` created from labeled data
- [ ] `input/images/sex_labels.csv` created from labeled data
- [ ] `input/images/orientation_labels.csv` created from labeled data
- [ ] `input/images/wing_labels.csv` created from labeled data
- [ ] `python validate_dataset.py` runs without [!] errors

**Current status:** No real data available. Pipeline runs in synthetic/fallback mode only.

---

## D. Model Evaluation Checklist

### FlyCount (Member 1)
- [x] Decision Tree implemented (depth 5, balanced, min_samples_leaf=5)
- [x] Rule-based fallback implemented and documented
- [x] Model saved to models/fly_count_model.joblib
- [x] Confusion matrix saved to output/confusion_matrix_fly_count.png
- [x] Decision tree diagram saved to output/tree_fly_count.png
- [ ] **Evaluation on real data: PENDING** (requires fly_count_labels.csv)
- [x] Synthetic accuracy reported separately and clearly labelled: 99.12% (SYNTHETIC ONLY)

### Wing Angle (Member 1)
- [x] HOG + PCA + Linear Regression implemented
- [x] Preprocessing pipeline (blur + CLAHE + wing split) implemented
- [x] Separate models for left and right wing
- [ ] **Evaluation on real data: PENDING** (requires wing_labels.csv)
- [x] Models saved: wing_angle_right_model.joblib, wing_angle_left_model.joblib
- [x] Scatter plots saved to output/ when real data is available

### Sex Classification (Member 2)
- [x] Logistic Regression + StandardScaler implemented
- [x] classify_pair() interface implemented
- [x] Data check strict — no fabrication
- [ ] **Model training: PENDING** (requires sex_labels.csv)
- [ ] **Evaluation: PENDING**

### Orientation (Member 2)
- [x] Stage 1: image moments baseline fully functional
- [x] 180° ambiguity documented clearly
- [x] Stage 2: HOG + PCA + LogReg implemented
- [ ] **Stage 2 training: PENDING** (requires orientation_labels.csv)
- [x] Visualization with ambiguity warning saved to output/

---

## E. README Checklist

- [x] Project title and problem statement
- [x] Team members and responsibilities
- [x] Installation instructions (pip install -r requirements.txt)
- [x] Training instructions for every module
- [x] Run instructions with example commands
- [x] Module descriptions with algorithms
- [x] What each output file means
- [x] HUD legend for annotated frames
- [x] Honest module status table
- [x] Limitations section
- [x] Dataset source (Dropbox link)
- [x] References

---

## F. PDF Report Checklist

- [ ] Title, authors, group number
- [ ] Problem statement (1 paragraph)
- [ ] Pipeline architecture diagram
- [ ] Description of each algorithm with justification
- [ ] Evaluation results — **use only real-data metrics; state "PENDING" if unavailable**
- [ ] Limitations
- [ ] Individual contributions
- [ ] References (CS229 paper, scikit-learn, OpenCV)
- [ ] Do NOT include synthetic accuracy as real accuracy

---

## G. Presentation / Demo Checklist

- [x] `python main.py` runs live demo on synthetic video (no data needed)
- [ ] Real video demo prepared if dataset downloaded
- [x] Module status HUD visible in output frames
- [x] FPS counter visible in output frames
- [x] N/A overlays shown for unavailable modules

---

## H. Individual Contribution Checklist

**Member 1 (Thrupthi)**
- [x] preprocessing.py (shared)
- [x] fly_count.py — Decision Tree classifier, rule-based fallback, visualization
- [x] wing_angle.py — HOG + PCA + Linear Regression, preprocessing pipeline
- [x] main.py — integration, FPS, error handling, annotate_frame
- [x] tests/ — full pytest suite
- [x] validate_dataset.py
- [x] PROJECT_AUDIT.md, SUBMISSION_CHECKLIST.md

**Member 2**
- [x] sex_classification.py — Logistic Regression, classify_pair()
- [x] orientation.py — moments baseline, HOG disambiguation

---

## I. Known Limitations

1. **No real data available** — all supervised models trained on synthetic distributions only. Real-world accuracy is unknown and must be evaluated after downloading the CS229 dataset.
2. **FlyCount zero-class** — the "zero" class is represented by noise blobs in synthetic data. Real video debris could cause false positives.
3. **Orientation 180° ambiguity** — Stage 1 (moments only) cannot distinguish head from abdomen. Stage 2 requires labeled patches.
4. **Wing angle** — model trained on synthetic patches. HOG features from real flies will differ significantly; retraining is required.
5. **No temporal tracking** — each frame is processed independently. Fly identity is not maintained across frames.
6. **Real-time claim** — measured at ~44–75 FPS on 320×240 synthetic video (Windows, CPU only). Real video at higher resolution will be slower. The pipeline is not verified as real-time on real data.
7. **Two-fly assumption** — the pipeline is designed for exactly 2 flies. Performance with 0 or more than 2 flies is untested.

---

## J. Remaining Tasks Requiring Human Input

| Task | Blocked by | Who |
|------|-----------|-----|
| Download CS229 dataset (~2 GB) | Manual download from Dropbox | Both members |
| Create fly_count_labels.csv from real contours | Dataset needed | Member 1 |
| Create sex_labels.csv from real contours | Dataset needed | Member 2 |
| Create orientation_labels.csv with flip annotations | Dataset needed + manual labeling | Member 2 |
| Create wing_labels.csv with angle annotations | Dataset needed + annotation tool | Member 1 |
| Retrain all supervised models on real data | All CSVs needed | Both |
| Report real evaluation metrics in PDF | Real training needed | Both |
| Test on real videos (test1.mp4–test5.mp4) | Dataset download needed | Both |

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo (no data needed)
python main.py

# Train FlyCount (synthetic fallback)
python fly_count.py

# Train with real data (after downloading dataset)
python fly_count.py --data input/images/fly_count_labels.csv
python wing_angle.py --data input/images/wing_labels.csv
python sex_classification.py --data input/images/sex_labels.csv
python orientation.py --data input/images/orientation_labels.csv

# Validate dataset
python validate_dataset.py

# Run all tests
pytest tests/ -v

# Run pipeline on real video
python main.py --video input/video/test4.mp4 --frames 100
```
