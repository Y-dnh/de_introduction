# Dataset Analysis Report

    This document contains a comprehensive analysis of the COCO-format dataset.

    ## 📊 Dataset Overview

    ### Available Splits
    - **TRAIN**: 22,500 images, 112,098 annotations, 3 categories
    - **VAL**: 6,915 images, 34,264 annotations, 3 categories
    
    ### Dataset Statistics
    - **Total Images**: 29,415
    - **Total Annotations**: 146,362
    - **Total Categories**: 3
    - **Dataset Path**: `coco_sama`

    ### Split Distribution
- **TRAIN**: 76.5% images, 76.6% annotations
- **VAL**: 23.5% images, 23.4% annotations

## 🏷️ Category Analysis

### TRAIN Split

| Category | Images | Annotations | Avg per Image |
|----------|--------|-------------|---------------|
| Person | 13,066 | 71,488 | 5.47 |
| Car | 8,299 | 31,862 | 3.84 |
| Pet | 6,899 | 8,748 | 1.27 |

### VAL Split

| Category | Images | Annotations | Avg per Image |
|----------|--------|-------------|---------------|
| Person | 3,983 | 22,344 | 5.61 |
| Car | 2,499 | 9,416 | 3.77 |
| Pet | 1,940 | 2,504 | 1.29 |

## 📐 Image Dimensions Analysis

### TRAIN Split

| Dimension | Min | Max | Mean | Median | Std Dev |
|-----------|-----|-----|------|--------|---------|
| Width (px) | 119 | 640 | 579 | 640 | 90 |
| Height (px) | 120 | 640 | 481 | 480 | 96 |
| Aspect Ratio | 0.24 | 4.38 | 1.26 | 1.33 | 0.33 |
| Area (px²) | 19,200 | 409,600 | 275,370 | 273,920 | 53,465 |

### VAL Split

| Dimension | Min | Max | Mean | Median | Std Dev |
|-----------|-----|-----|------|--------|---------|
| Width (px) | 150 | 640 | 580 | 640 | 91 |
| Height (px) | 120 | 640 | 478 | 480 | 95 |
| Aspect Ratio | 0.34 | 4.81 | 1.27 | 1.33 | 0.34 |
| Area (px²) | 19,200 | 409,600 | 274,518 | 273,920 | 53,606 |

## 🔍 Annotation Analysis

### TRAIN Split

#### Bounding Box Statistics
| Metric | Min | Max | Mean | Median | Std Dev |
|--------|-----|-----|------|--------|---------|
| BBox Width | 1 | 640 | 83 | 37 | 111 |
| BBox Height | 2 | 640 | 101 | 52 | 114 |
| BBox Area | 2 | 408321 | 9912 | 966 | 25694 |
| BBox Aspect Ratio | 0.05 | 32.38 | 0.97 | 0.73 | 0.83 |
| Annotations per Image | 0 | 308 | 4.98 | 2.00 | 9.09 |

### VAL Split

#### Bounding Box Statistics
| Metric | Min | Max | Mean | Median | Std Dev |
|--------|-----|-----|------|--------|---------|
| BBox Width | 2 | 640 | 82 | 37 | 111 |
| BBox Height | 2 | 640 | 102 | 53 | 113 |
| BBox Area | 2 | 334872 | 9846 | 990 | 25852 |
| BBox Aspect Ratio | 0.04 | 22.20 | 0.95 | 0.71 | 0.82 |
| Annotations per Image | 0 | 185 | 4.96 | 2.00 | 8.55 |

## 💾 Storage Analysis

### TRAIN Split
- **Total Size**: 3.41 GB (3496 MB)
- **File Count**: 22,500
- **Average File Size**: 0.16 MB
- **Size Range**: 8 KB - 0.55 MB

### VAL Split
- **Total Size**: 1.05 GB (1075 MB)
- **File Count**: 6,915
- **Average File Size**: 0.16 MB
- **Size Range**: 6 KB - 1.86 MB

### Total Dataset Storage
- **Total Size**: 4.46 GB
- **Total Files**: 29,415

## 🎭 Scene Complexity Analysis

### TRAIN Split

#### Complexity Distribution
- **Simple Scenes** (0-1 objects): 9,085 images (40.4%)
- **Moderate Scenes** (2-3 objects): 5,395 images (24.0%)
- **Complex Scenes** (4+ objects): 8,020 images (35.6%)
- **Average Objects per Image**: 4.98

#### Detailed Object Count Distribution
- **0 objects**: 3,000 images (13.3%)
- **1 objects**: 6,085 images (27.0%)
- **2 objects**: 3,464 images (15.4%)
- **3 objects**: 1,931 images (8.6%)
- **4 objects**: 1,265 images (5.6%)
- **5 objects**: 961 images (4.3%)
- **6 objects**: 819 images (3.6%)
- **7 objects**: 648 images (2.9%)
- **8 objects**: 555 images (2.5%)
- **9 objects**: 460 images (2.0%)

### VAL Split

#### Complexity Distribution
- **Simple Scenes** (0-1 objects): 2,920 images (42.2%)
- **Moderate Scenes** (2-3 objects): 1,558 images (22.5%)
- **Complex Scenes** (4+ objects): 2,437 images (35.2%)
- **Average Objects per Image**: 4.96

#### Detailed Object Count Distribution
- **0 objects**: 1,000 images (14.5%)
- **1 objects**: 1,920 images (27.8%)
- **2 objects**: 985 images (14.2%)
- **3 objects**: 573 images (8.3%)
- **4 objects**: 355 images (5.1%)
- **5 objects**: 300 images (4.3%)
- **6 objects**: 230 images (3.3%)
- **7 objects**: 189 images (2.7%)
- **8 objects**: 191 images (2.8%)
- **9 objects**: 132 images (1.9%)

## 📊 Visualizations

The analysis includes comprehensive visualizations showing:

1. **Category Distribution** - Image and annotation counts per category
2. **Image Dimensions** - Width vs height scatter plot
3. **Aspect Ratios** - Distribution of image aspect ratios
4. **Bounding Box Sizes** - Distribution of annotation dimensions
5. **Annotations per Image** - Scene complexity histogram
6. **File Sizes** - Storage requirements distribution
7. **Scene Complexity** - Object count per image
8. **Category Annotations** - Detailed annotation counts
9. **Bounding Box Areas** - Size distribution of objects

## 📁 Dataset Structure

```
coco_sama/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
├── train/
├── val/
├── dataset_analysis_*.png
├── analysis_results_*.json
└── README.md
```

---

*This analysis was automatically generated by the Dataset Analyzer*

*Generated on: 2025-07-16 15:28:29*
