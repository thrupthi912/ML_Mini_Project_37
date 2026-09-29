"""Investigate why single blob on plain background gives 0 contours."""
import cv2
import numpy as np
from preprocessing import threshold_frame, extract_contours

frame_one = np.full((240, 320), 200, dtype=np.uint8)
cv2.ellipse(frame_one, (160, 120), (12, 7), 0, 0, 360, 30, -1)

# Without background (background=None)
mask_no_bg = threshold_frame(frame_one, background=None)
c_no_bg = extract_contours(mask_no_bg)
print(f"No background: mask unique values={np.unique(mask_no_bg)}, contours={len(c_no_bg)}")

# With matching background (same as frame but without the blob)
bg = np.full((240, 320), 200, dtype=np.uint8)
mask_with_bg = threshold_frame(frame_one, background=bg)
c_with_bg = extract_contours(mask_with_bg)
print(f"With background: mask unique values={np.unique(mask_with_bg)}, contours={len(c_with_bg)}")
if c_with_bg:
    import cv2 as cv2_
    print(f"  Areas: {[cv2_.contourArea(c) for c in c_with_bg]}")

# Threshold value chosen by Otsu on the diff image
diff = cv2.absdiff(frame_one, bg)
blurred = cv2.GaussianBlur(diff, (5, 5), 0)
thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold on diff: {thresh_val}")

# The problem: without background, Otsu on the raw frame picks a threshold that
# binarizes the whole frame because the histogram is bimodal (dark ellipse + light bg)
# but the absdiff is near-zero everywhere except the blob.
thresh_val2, _ = cv2.threshold(
    cv2.GaussianBlur(frame_one, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold on raw frame (no bg): {thresh_val2}")
# When no background is given, the raw frame IS already bimodal → this should work
# But min_area=50 might be filtering it out — let's check
mask_raw = threshold_frame(frame_one, background=None)
all_contours, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"All contours before area filter: {len(all_contours)}")
for c in all_contours:
    print(f"  area={cv2.contourArea(c):.1f}")
