# Dataset Analysis Report

    This document contains a comprehensive analysis of the COCO-format dataset.

    ## 📊 Dataset Overview

    ### Available Splits
    - **TRAIN**: 22,500 images, 94,209 annotations, 3 categories
    - **VAL**: 6,789 images, 27,861 annotations, 3 categories
    
    ### Dataset Statistics
    - **Total Images**: 29,289
    - **Total Annotations**: 122,070
    - **Total Categories**: 3
    - **Dataset Path**: `coco`

    ### Split Distribution
- **TRAIN**: 76.8% images, 77.2% annotations
- **VAL**: 23.2% images, 22.8% annotations

## 🏷️ Category Analysis

### TRAIN Split

| Category | Images | Annotations | Avg per Image |
|----------|--------|-------------|---------------|
| Person | 12,780 | 55,519 | 4.34 |
| Car | 8,260 | 30,233 | 3.66 |
| Pet | 6,836 | 8,457 | 1.24 |

### VAL Split

| Category | Images | Annotations | Avg per Image |
|----------|--------|-------------|---------------|
| Person | 3,870 | 16,704 | 4.32 |
| Car | 2,455 | 8,918 | 3.63 |
| Pet | 1,804 | 2,239 | 1.24 |

## 📐 Image Dimensions Analysis

### TRAIN Split

| Dimension | Min | Max | Mean | Median | Std Dev |
|-----------|-----|-----|------|--------|---------|
| Width (px) | 59 | 640 | 580 | 640 | 90 |
| Height (px) | 72 | 640 | 479 | 480 | 95 |
| Aspect Ratio | 0.30 | 4.81 | 1.27 | 1.33 | 0.34 |
| Area (px²) | 4,248 | 409,600 | 274,987 | 273,920 | 53,692 |

### VAL Split

| Dimension | Min | Max | Mean | Median | Std Dev |
|-----------|-----|-----|------|--------|---------|
| Width (px) | 160 | 640 | 579 | 640 | 91 |
| Height (px) | 120 | 640 | 480 | 480 | 95 |
| Aspect Ratio | 0.34 | 3.88 | 1.26 | 1.33 | 0.33 |
| Area (px²) | 19,200 | 409,600 | 274,874 | 273,920 | 52,332 |

## 🔍 Annotation Analysis

### TRAIN Split

#### Bounding Box Statistics
| Metric | Min | Max | Mean | Median | Std Dev |
|--------|-----|-----|------|--------|---------|
| BBox Width | 1 | 640 | 95 | 44 | 123 |
| BBox Height | 1 | 640 | 108 | 59 | 118 |
| BBox Area | 1 | 396810 | 11558 | 1377 | 26975 |
| BBox Aspect Ratio | 0.04 | 40.09 | 1.03 | 0.76 | 0.97 |
| Annotations per Image | 0 | 54 | 4.19 | 2.00 | 5.04 |

### VAL Split

#### Bounding Box Statistics
| Metric | Min | Max | Mean | Median | Std Dev |
|--------|-----|-----|------|--------|---------|
| BBox Width | 1 | 640 | 96 | 44 | 124 |
| BBox Height | 2 | 640 | 110 | 60 | 119 |
| BBox Area | 2 | 306631 | 11847 | 1408 | 27405 |
| BBox Aspect Ratio | 0.06 | 40.09 | 1.02 | 0.75 | 0.99 |
| Annotations per Image | 0 | 42 | 4.10 | 2.00 | 4.99 |

## 💾 Storage Analysis

### TRAIN Split
- **Total Size**: 3.41 GB (3490 MB)
- **File Count**: 22,500
- **Average File Size**: 0.16 MB
- **Size Range**: 6 KB - 1.86 MB

### VAL Split
- **Total Size**: 1.03 GB (1059 MB)
- **File Count**: 6,789
- **Average File Size**: 0.16 MB
- **Size Range**: 6 KB - 0.49 MB

### Total Dataset Storage
- **Total Size**: 4.44 GB
- **Total Files**: 29,289

## 🎭 Scene Complexity Analysis

### TRAIN Split

#### Complexity Distribution
- **Simple Scenes** (0-1 objects): 9,440 images (42.0%)
- **Moderate Scenes** (2-3 objects): 5,193 images (23.1%)
- **Complex Scenes** (4+ objects): 7,867 images (35.0%)
- **Average Objects per Image**: 4.19

#### Detailed Object Count Distribution
- **0 objects**: 3,000 images (13.3%)
- **1 objects**: 6,440 images (28.6%)
- **2 objects**: 3,387 images (15.1%)
- **3 objects**: 1,806 images (8.0%)
- **4 objects**: 1,164 images (5.2%)
- **5 objects**: 961 images (4.3%)
- **6 objects**: 717 images (3.2%)
- **7 objects**: 629 images (2.8%)
- **8 objects**: 540 images (2.4%)
- **9 objects**: 470 images (2.1%)

### VAL Split

#### Complexity Distribution
- **Simple Scenes** (0-1 objects): 2,967 images (43.7%)
- **Moderate Scenes** (2-3 objects): 1,484 images (21.9%)
- **Complex Scenes** (4+ objects): 2,338 images (34.4%)
- **Average Objects per Image**: 4.10

#### Detailed Object Count Distribution
- **0 objects**: 1,000 images (14.7%)
- **1 objects**: 1,967 images (29.0%)
- **2 objects**: 960 images (14.1%)
- **3 objects**: 524 images (7.7%)
- **4 objects**: 346 images (5.1%)
- **5 objects**: 292 images (4.3%)
- **6 objects**: 208 images (3.1%)
- **7 objects**: 164 images (2.4%)
- **8 objects**: 162 images (2.4%)
- **9 objects**: 149 images (2.2%)

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
coco/
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

*Generated on: 2025-07-16 15:28:05*
