# Engineered Features Description

This document describes the 38 engineered features extracted from the raw steering images for classification.

---

## Overview

The features are organized into **5 categories**:

| Category | Count | Purpose |
|----------|-------|---------|
| Edge-Based | 5 | Detect edges, lane markings, road boundaries |
| Spatial/Region | 11 | Capture where brightness is located (critical for steering) |
| Texture | 7 | Describe surface patterns using LBP and GLCM |
| Histogram/Distribution | 11 | Overall pixel intensity distribution |
| Gradient/HOG | 4 | Shape and edge direction information |

---

## 1. Edge-Based Features (5 features)

These features capture edge information using gradient operators, useful for detecting lane markings and road boundaries.

| Feature | How It's Computed | What It Tells Us |
|---------|-------------------|------------------|
| `edge_mean` | Mean of Sobel edge magnitude (√(Gx² + Gy²)) | Overall edge intensity in the image; higher = more edges/details |
| `edge_std` | Standard deviation of edge magnitude | Variation in edge strength; higher = more heterogeneous edges |
| `edge_max` | Maximum edge magnitude | Strongest edge in the image |
| `edge_hv_ratio` | Horizontal edge energy / Vertical edge energy | Ratio > 1 means more horizontal edges (e.g., road horizon); < 1 means more vertical (e.g., lane lines) |
| `canny_density` | Fraction of pixels detected as edges by Canny | Overall "edginess" of the image; low = smooth scene |

---

## 2. Spatial/Region Features (11 features)

These capture **where** brightness is located in the image — critical for steering direction prediction.

| Feature | How It's Computed | What It Tells Us |
|---------|-------------------|------------------|
| `lr_asymmetry` | Mean(left half) - Mean(right half) | **Key feature!** Positive = left side brighter (suggests left turn) |
| `lr_ratio` | Mean(left half) / Mean(right half) | Similar to above, but as a ratio |
| `tb_ratio` | Mean(top half) / Mean(bottom half) | Sky vs road brightness; higher = brighter sky |
| `tb_diff` | Mean(top half) - Mean(bottom half) | Absolute brightness difference top vs bottom |
| `quad_tl_mean` | Mean intensity of top-left quadrant | Brightness in upper-left region |
| `quad_tr_mean` | Mean intensity of top-right quadrant | Brightness in upper-right region |
| `quad_bl_mean` | Mean intensity of bottom-left quadrant | Brightness in lower-left region |
| `quad_br_mean` | Mean intensity of bottom-right quadrant | Brightness in lower-right region |
| `diag_asymmetry` | (TL + BR) - (TR + BL) | Diagonal brightness pattern |
| `brightness_center_x` | Weighted centroid X (normalized to [-1, 1]) | **Key feature!** Negative = brightness center is left; positive = right |
| `brightness_center_y` | Weighted centroid Y (normalized to [-1, 1]) | Negative = brightness center is up; positive = down |

---

## 3. Texture Features (7 features)

These describe the surface texture patterns using Local Binary Patterns (LBP) and Gray Level Co-occurrence Matrix (GLCM).

### Local Binary Pattern (LBP)

LBP compares each pixel to its neighbors, encoding local texture as a binary pattern.

| Feature | How It's Computed | What It Tells Us |
|---------|-------------------|------------------|
| `lbp_mean` | Mean of LBP image | Average local texture pattern code |
| `lbp_std` | Std deviation of LBP values | Texture pattern variation |
| `lbp_uniformity` | Sum of squared LBP histogram bins | Higher = more uniform texture (fewer distinct patterns) |

### Gray Level Co-occurrence Matrix (GLCM)

GLCM analyzes how often pairs of pixels with specific values occur at specific spatial relationships.

| Feature | How It's Computed | What It Tells Us |
|---------|-------------------|------------------|
| `glcm_contrast` | GLCM contrast (averaged over 4 angles) | Local intensity variation; higher = rougher texture |
| `glcm_correlation` | GLCM correlation | How correlated a pixel is with its neighbors; higher = more structured patterns |
| `glcm_energy` | GLCM energy (angular second moment) | Texture uniformity; higher = more regular/repetitive |
| `glcm_homogeneity` | GLCM homogeneity | Closeness of distribution to diagonal; higher = smoother transitions |

---

## 4. Histogram/Distribution Features (11 features)

These capture the overall pixel intensity distribution.

| Feature | How It's Computed | What It Tells Us |
|---------|-------------------|------------------|
| `intensity_mean` | Mean pixel value | Overall image brightness |
| `intensity_std` | Std deviation of pixel values | Image contrast; higher = more variation |
| `percentile_10` | 10th percentile of pixel values | Darkest regions (shadows) |
| `percentile_25` | 25th percentile (Q1) | Lower brightness bound |
| `percentile_50` | Median pixel value | Central brightness |
| `percentile_75` | 75th percentile (Q3) | Upper brightness bound |
| `percentile_90` | 90th percentile | Brightest regions (highlights) |
| `iqr` | Q3 - Q1 (interquartile range) | Spread of middle 50% of intensities |
| `skewness` | Skewness of pixel distribution | Negative = tail toward dark; Positive = tail toward bright |
| `kurtosis` | Kurtosis of pixel distribution | Peakedness; high = concentrated around mean; low = flat distribution |
| `intensity_range` | Max - Min pixel value | Dynamic range of the image |

---

## 5. Gradient/HOG Features (4 features)

Summary statistics from Histogram of Oriented Gradients (HOG), which captures shape and edge direction information.

HOG divides the image into cells, computes gradient orientations within each cell, and creates histograms of these orientations. The full HOG descriptor has 1764 values, so we summarize it with statistics.

| Feature | How It's Computed | What It Tells Us |
|---------|-------------------|------------------|
| `hog_mean` | Mean of full HOG descriptor (1764 values) | Average gradient orientation strength |
| `hog_std` | Std deviation of HOG values | Variation in gradient orientations |
| `hog_max` | Maximum HOG bin value | Strongest gradient direction |
| `hog_energy` | Sum of squared HOG values | Total gradient energy; higher = more edges/structure |

---

## Most Discriminative Features for Steering

Based on ANOVA F-scores and Mutual Information analysis, the **most important features for predicting steering direction** are:

### Top Features

1. **`lr_asymmetry`** — Left-right brightness difference is directly related to steering direction
2. **`brightness_center_x`** — Where the "center of mass" of brightness is located horizontally
3. **`quad_bl_mean`** / **`quad_br_mean`** — Bottom quadrant brightness patterns
4. **`lr_ratio`** — Ratio version of left-right asymmetry
5. **`edge_hv_ratio`** — Balance of horizontal vs vertical edges

### Why These Features Work

These spatial features make intuitive sense for steering prediction:

- **Left turns**: The left side of the road typically shows different features (obstacles, curves, lane markings) than the right side, causing asymmetry in brightness.
- **Right turns**: The opposite pattern occurs.
- **Forward/Straight**: Both sides tend to be more balanced.

The `brightness_center_x` feature essentially asks: "Where is the visual 'weight' of the image?" If it's shifted left, the vehicle may need to turn left to center the road.

---

## Feature Extraction Methods

### Sobel Edge Detection
Computes image gradients using convolution with Sobel kernels to find edges in horizontal and vertical directions.

### Canny Edge Detection
Multi-stage algorithm that detects a wide range of edges using gradient magnitude and non-maximum suppression.

### Local Binary Pattern (LBP)
Texture descriptor that labels pixels by thresholding the neighborhood around each pixel and encoding the result as a binary number.

### Gray Level Co-occurrence Matrix (GLCM)
Statistical method that considers the spatial relationship of pixels, computing how often pairs of pixels with specific gray-level values occur.

### Histogram of Oriented Gradients (HOG)
Feature descriptor that counts occurrences of gradient orientation in localized portions of an image, commonly used for object detection.

---

## Usage

These features are saved in `data/engineered_features.csv` with 9,900 samples (one per image) and 38 features plus the label column.

```python
import pandas as pd

# Load engineered features
df = pd.read_csv('data/engineered_features.csv')

# Separate features and labels
feature_cols = [col for col in df.columns if col != 'label']
X = df[feature_cols].values
y = df['label'].values
```

