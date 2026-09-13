# EN3160 - Assignment 01: Intensity Transformations and Neighborhood Filtering

**Course:** EN3160 - Image Processing and Machine Vision
**Index No:** 230186E
**University of Moratuwa**

## Overview

This repository contains the solutions for Assignment 01, covering fundamental image processing techniques including intensity transformations, histogram equalization, spatial filtering, image zooming, segmentation, and edge-preserving smoothing.

All questions are implemented in Python using NumPy, OpenCV, and Matplotlib. Most core operations (intensity mapping, filtering, interpolation) are implemented from scratch to demonstrate understanding of the underlying algorithms, alongside comparisons with their corresponding OpenCV built-in functions where required.

## Repository Structure

```
.
├── EN3160_A01.ipynb              # Main notebook with all questions, plots, and discussion
├── Individual code scripts/       # Standalone .py script for each question
│   ├── Q01.py                     # Piecewise-linear intensity transformation
│   ├── Q02.py                     # White/gray matter accentuation (brain PD image)
│   ├── Q03.py                     # Vibrance enhancement (HSV saturation transform)
│   ├── Q04.py                     # Histogram equalization (from scratch)
│   ├── Q05.py                     # (if applicable, adjust to your actual numbering)
│   ├── Q06.py                     # Foreground-only histogram equalization
│   ├── Q07.py                     # Sobel filtering (filter2D, manual, separable)
│   ├── Q08.py                     # Image zoom (nearest-neighbor & bilinear) + SSD
│   ├── Q09.py                     # GrabCut segmentation + background blur
│   └── Q10.py                     # Bilateral filtering (OpenCV vs. own implementation)
├── images/                        # Input images used across the questions
├── results/                       # Saved output plots/figures
└── README.md
```

> Adjust the file/question numbering above to match your actual folder if it differs.

## Questions Covered

| # | Topic | Key Techniques |
|---|-------|-----------------|
| 1 | Piecewise-linear intensity transformation | Custom LUT-based transform, breakpoint experimentation |
| 2 | White/gray matter accentuation | Intensity windowing on a brain proton density image |
| 3 | Vibrance enhancement | Gaussian-weighted saturation boost on the HSV S-plane |
| 4 | Histogram equalization | CDF-based equalization implemented from scratch |
| 5 | Foreground-only histogram equalization | HSV thresholding, masking, selective equalization |
| 6 | Sobel edge detection | `cv2.filter2D`, manual convolution, separable kernel decomposition |
| 7 | Image zoom | Nearest-neighbor & bilinear interpolation, normalized SSD validation |
| 8 | Image segmentation | `cv2.grabCut`, background blur (bokeh effect) |
| 9 | Bilateral filtering | `cv2.bilateralFilter` vs. custom implementation, SSD comparison |

## Running the Code

### Requirements

```bash
pip install numpy opencv-python matplotlib scipy
```

### Option 1 — Jupyter Notebook

Open `EN3160_A01.ipynb` and run all cells. Each question is organized in its own section with code, output plots, and a discussion of results.

### Option 2 — Individual Scripts

Each question can also be run independently:

```bash
cd "Individual code scripts"
python Q01.py
```

Make sure the `images/` folder is present relative to wherever the script is run from, since all scripts load input images using relative paths (e.g. `images/q1.jpg`).

## Notes

- Where OpenCV provides a built-in function (e.g. `filter2D`, `bilateralFilter`, `equalizeHist`), a manual from-scratch implementation is also included and compared quantitatively (typically via normalized SSD) to verify correctness.
- Breakpoints, thresholds, and filter parameters (σ, kernel size, etc.) were tuned per-image and are documented alongside each result.
