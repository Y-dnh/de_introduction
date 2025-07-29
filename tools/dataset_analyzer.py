import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from pycocotools.coco import COCO
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install with: pip install pycocotools tqdm")
    exit(1)


class DatasetAnalyzer:
    """Comprehensive dataset analyzer for COCO-format datasets with flexible split support"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / "annotations"

        # Initialize available splits
        self.available_splits = {}
        self.coco_objects = {}
        self.image_paths = {}

        # Discover available annotation files and corresponding image directories
        self._discover_splits()

        # Analysis results storage
        self.analysis_results = {}

        print(f"Dataset loaded from: {dataset_path}")
        print(f"Available splits: {list(self.available_splits.keys())}")
        for split, info in self.available_splits.items():
            print(f"  {split}: {info['images']} images, {info['annotations']} annotations")

    def _discover_splits(self):
        """Discover available splits in the dataset"""
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_path}")

        # Common annotation file patterns
        annotation_patterns = [
            "instances_train.json", "instances_val.json", "instances_test.json",
            "train.json", "val.json", "test.json",
            "instances_train2017.json", "instances_val2017.json", "instances_test2017.json"
        ]

        # Find all JSON files in annotations directory
        annotation_files = list(self.annotations_path.glob("*.json"))

        for ann_file in annotation_files:
            # Extract split name from filename
            filename = ann_file.name

            # Try to extract split name
            split_name = None
            if filename.startswith("instances_"):
                split_name = filename.replace("instances_", "").replace(".json", "")
            elif filename.endswith(".json"):
                split_name = filename.replace(".json", "")

            if split_name:
                try:
                    # Load COCO object
                    coco_obj = COCO(str(ann_file))
                    self.coco_objects[split_name] = coco_obj

                    # Try to find corresponding image directory
                    possible_image_dirs = [
                        self.dataset_path / split_name,
                        self.dataset_path / f"{split_name}2017",
                        self.dataset_path / "images" / split_name,
                        self.dataset_path / "images" / f"{split_name}2017"
                    ]

                    image_dir = None
                    for img_dir in possible_image_dirs:
                        if img_dir.exists():
                            image_dir = img_dir
                            break

                    self.image_paths[split_name] = image_dir

                    # Store split info
                    self.available_splits[split_name] = {
                        'images': len(coco_obj.getImgIds()),
                        'annotations': len(coco_obj.getAnnIds()),
                        'categories': len(coco_obj.getCatIds()),
                        'annotation_file': ann_file,
                        'image_dir': image_dir
                    }

                except Exception as e:
                    print(f"Warning: Could not load {ann_file}: {e}")
                    continue

        if not self.available_splits:
            raise ValueError("No valid COCO annotation files found in the dataset")

    def analyze_basic_stats(self):
        """Analyze basic dataset statistics for all available splits"""
        print("\n=== ANALYZING BASIC STATISTICS ===")

        stats = {}

        for split_name, coco_obj in self.coco_objects.items():
            stats[split_name] = {
                'images': len(coco_obj.getImgIds()),
                'annotations': len(coco_obj.getAnnIds()),
                'categories': len(coco_obj.getCatIds())
            }

        # Calculate totals
        stats['total'] = {
            'images': sum(split_stats['images'] for split_stats in stats.values()),
            'annotations': sum(split_stats['annotations'] for split_stats in stats.values()),
            'categories': len(set().union(*[coco_obj.getCatIds() for coco_obj in self.coco_objects.values()]))
        }

        self.analysis_results['basic_stats'] = stats
        return stats

    def analyze_categories(self, splits: Optional[List[str]] = None):
        """Analyze category distribution for specified splits"""
        print("\n=== ANALYZING CATEGORIES ===")

        if splits is None:
            splits = list(self.coco_objects.keys())

        categories_info = {}

        for split_name in splits:
            if split_name not in self.coco_objects:
                print(f"Warning: Split '{split_name}' not found, skipping...")
                continue

            coco_obj = self.coco_objects[split_name]
            cat_stats = {}

            for cat_id in coco_obj.getCatIds():
                cat_info = coco_obj.loadCats([cat_id])[0]
                cat_name = cat_info['name']

                # Get images and annotations for this category
                img_ids = coco_obj.getImgIds(catIds=[cat_id])
                ann_ids = coco_obj.getAnnIds(catIds=[cat_id])

                cat_stats[cat_name] = {
                    'id': cat_id,
                    'images': len(img_ids),
                    'annotations': len(ann_ids),
                    'avg_annotations_per_image': len(ann_ids) / len(img_ids) if img_ids else 0
                }

            categories_info[split_name] = cat_stats

        self.analysis_results['categories'] = categories_info
        return categories_info

    def analyze_image_dimensions(self, splits: Optional[List[str]] = None):
        """Analyze image dimensions and aspect ratios for specified splits"""
        print("\n=== ANALYZING IMAGE DIMENSIONS ===")

        if splits is None:
            splits = list(self.coco_objects.keys())

        dimension_stats = {}

        for split_name in splits:
            if split_name not in self.coco_objects:
                print(f"Warning: Split '{split_name}' not found, skipping...")
                continue

            coco_obj = self.coco_objects[split_name]
            widths = []
            heights = []
            aspect_ratios = []
            areas = []

            for img_id in tqdm(coco_obj.getImgIds(), desc=f"Analyzing {split_name} images"):
                img_info = coco_obj.loadImgs([img_id])[0]
                width = img_info['width']
                height = img_info['height']

                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)
                areas.append(width * height)

            dimension_stats[split_name] = {
                'width': {
                    'min': min(widths) if widths else 0,
                    'max': max(widths) if widths else 0,
                    'mean': np.mean(widths) if widths else 0,
                    'median': np.median(widths) if widths else 0,
                    'std': np.std(widths) if widths else 0
                },
                'height': {
                    'min': min(heights) if heights else 0,
                    'max': max(heights) if heights else 0,
                    'mean': np.mean(heights) if heights else 0,
                    'median': np.median(heights) if heights else 0,
                    'std': np.std(heights) if heights else 0
                },
                'aspect_ratio': {
                    'min': min(aspect_ratios) if aspect_ratios else 0,
                    'max': max(aspect_ratios) if aspect_ratios else 0,
                    'mean': np.mean(aspect_ratios) if aspect_ratios else 0,
                    'median': np.median(aspect_ratios) if aspect_ratios else 0,
                    'std': np.std(aspect_ratios) if aspect_ratios else 0
                },
                'area': {
                    'min': min(areas) if areas else 0,
                    'max': max(areas) if areas else 0,
                    'mean': np.mean(areas) if areas else 0,
                    'median': np.median(areas) if areas else 0,
                    'std': np.std(areas) if areas else 0
                },
                'raw_data': {
                    'widths': widths,
                    'heights': heights,
                    'aspect_ratios': aspect_ratios,
                    'areas': areas
                }
            }

        self.analysis_results['dimensions'] = dimension_stats
        return dimension_stats

    def analyze_annotations(self, splits: Optional[List[str]] = None):
        """Analyze annotation properties for specified splits"""
        print("\n=== ANALYZING ANNOTATIONS ===")

        if splits is None:
            splits = list(self.coco_objects.keys())

        annotation_stats = {}

        for split_name in splits:
            if split_name not in self.coco_objects:
                print(f"Warning: Split '{split_name}' not found, skipping...")
                continue

            coco_obj = self.coco_objects[split_name]
            bbox_widths = []
            bbox_heights = []
            bbox_areas = []
            bbox_aspect_ratios = []
            annotations_per_image = []

            # Analyze annotations per image
            for img_id in tqdm(coco_obj.getImgIds(), desc=f"Analyzing {split_name} annotations"):
                ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
                annotations_per_image.append(len(ann_ids))

                if ann_ids:
                    anns = coco_obj.loadAnns(ann_ids)
                    for ann in anns:
                        bbox = ann['bbox']  # [x, y, width, height]
                        width, height = bbox[2], bbox[3]
                        area = ann['area']

                        bbox_widths.append(width)
                        bbox_heights.append(height)
                        bbox_areas.append(area)
                        bbox_aspect_ratios.append(width / height if height > 0 else 0)

            annotation_stats[split_name] = {
                'bbox_width': {
                    'min': min(bbox_widths) if bbox_widths else 0,
                    'max': max(bbox_widths) if bbox_widths else 0,
                    'mean': np.mean(bbox_widths) if bbox_widths else 0,
                    'median': np.median(bbox_widths) if bbox_widths else 0,
                    'std': np.std(bbox_widths) if bbox_widths else 0
                },
                'bbox_height': {
                    'min': min(bbox_heights) if bbox_heights else 0,
                    'max': max(bbox_heights) if bbox_heights else 0,
                    'mean': np.mean(bbox_heights) if bbox_heights else 0,
                    'median': np.median(bbox_heights) if bbox_heights else 0,
                    'std': np.std(bbox_heights) if bbox_heights else 0
                },
                'bbox_area': {
                    'min': min(bbox_areas) if bbox_areas else 0,
                    'max': max(bbox_areas) if bbox_areas else 0,
                    'mean': np.mean(bbox_areas) if bbox_areas else 0,
                    'median': np.median(bbox_areas) if bbox_areas else 0,
                    'std': np.std(bbox_areas) if bbox_areas else 0
                },
                'bbox_aspect_ratio': {
                    'min': min(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
                    'max': max(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
                    'mean': np.mean(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
                    'median': np.median(bbox_aspect_ratios) if bbox_aspect_ratios else 0,
                    'std': np.std(bbox_aspect_ratios) if bbox_aspect_ratios else 0
                },
                'annotations_per_image': {
                    'min': min(annotations_per_image) if annotations_per_image else 0,
                    'max': max(annotations_per_image) if annotations_per_image else 0,
                    'mean': np.mean(annotations_per_image) if annotations_per_image else 0,
                    'median': np.median(annotations_per_image) if annotations_per_image else 0,
                    'std': np.std(annotations_per_image) if annotations_per_image else 0
                },
                'raw_data': {
                    'bbox_widths': bbox_widths,
                    'bbox_heights': bbox_heights,
                    'bbox_areas': bbox_areas,
                    'bbox_aspect_ratios': bbox_aspect_ratios,
                    'annotations_per_image': annotations_per_image
                }
            }

        self.analysis_results['annotations'] = annotation_stats
        return annotation_stats

    def analyze_file_sizes(self, splits: Optional[List[str]] = None):
        """Analyze file sizes and disk usage for specified splits"""
        print("\n=== ANALYZING FILE SIZES ===")

        if splits is None:
            splits = list(self.coco_objects.keys())

        file_stats = {}

        for split_name in splits:
            if split_name not in self.image_paths or not self.image_paths[split_name]:
                print(f"Warning: Image directory not found for split '{split_name}', skipping file size analysis...")
                continue

            images_path = self.image_paths[split_name]

            if not images_path.exists():
                print(f"Warning: Image directory does not exist: {images_path}")
                continue

            file_sizes = []
            total_size = 0

            for img_file in tqdm(images_path.iterdir(), desc=f"Analyzing {split_name} file sizes"):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    size = img_file.stat().st_size
                    file_sizes.append(size)
                    total_size += size

            file_stats[split_name] = {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'total_size_gb': total_size / (1024 * 1024 * 1024),
                'avg_file_size_bytes': np.mean(file_sizes) if file_sizes else 0,
                'avg_file_size_mb': np.mean(file_sizes) / (1024 * 1024) if file_sizes else 0,
                'min_file_size_bytes': min(file_sizes) if file_sizes else 0,
                'max_file_size_bytes': max(file_sizes) if file_sizes else 0,
                'file_count': len(file_sizes),
                'raw_data': file_sizes
            }

        self.analysis_results['file_sizes'] = file_stats
        return file_stats

    def analyze_complexity_distribution(self, splits: Optional[List[str]] = None):
        """Analyze scene complexity for specified splits"""
        print("\n=== ANALYZING SCENE COMPLEXITY ===")

        if splits is None:
            splits = list(self.coco_objects.keys())

        complexity_stats = {}

        for split_name in splits:
            if split_name not in self.coco_objects:
                print(f"Warning: Split '{split_name}' not found, skipping...")
                continue

            coco_obj = self.coco_objects[split_name]
            complexity_distribution = Counter()

            for img_id in tqdm(coco_obj.getImgIds(), desc=f"Analyzing {split_name} complexity"):
                ann_ids = coco_obj.getAnnIds(imgIds=[img_id])
                num_objects = len(ann_ids)
                complexity_distribution[num_objects] += 1

            # Calculate complexity categories
            simple_scenes = sum(count for objects, count in complexity_distribution.items() if objects <= 1)
            moderate_scenes = sum(count for objects, count in complexity_distribution.items() if 2 <= objects <= 3)
            complex_scenes = sum(count for objects, count in complexity_distribution.items() if objects >= 4)

            total_images = sum(complexity_distribution.values())

            complexity_stats[split_name] = {
                'distribution': dict(complexity_distribution),
                'simple_scenes': simple_scenes,
                'moderate_scenes': moderate_scenes,
                'complex_scenes': complex_scenes,
                'simple_percentage': (simple_scenes / total_images) * 100 if total_images > 0 else 0,
                'moderate_percentage': (moderate_scenes / total_images) * 100 if total_images > 0 else 0,
                'complex_percentage': (complex_scenes / total_images) * 100 if total_images > 0 else 0,
                'avg_objects_per_image': np.mean([objects for objects, count in complexity_distribution.items()
                                                  for _ in range(count)]) if complexity_distribution else 0
            }

        self.analysis_results['complexity'] = complexity_stats
        return complexity_stats

    def analyze_single_split(self, split_name: str):
        """Analyze a single split separately"""
        print(f"\n=== ANALYZING SINGLE SPLIT: {split_name.upper()} ===")

        if split_name not in self.coco_objects:
            print(f"Error: Split '{split_name}' not found in available splits: {list(self.coco_objects.keys())}")
            return None

        # Run all analyses for this single split
        results = {}

        # Basic stats
        coco_obj = self.coco_objects[split_name]
        results['basic_stats'] = {
            'images': len(coco_obj.getImgIds()),
            'annotations': len(coco_obj.getAnnIds()),
            'categories': len(coco_obj.getCatIds())
        }

        # Other analyses
        results['categories'] = self.analyze_categories([split_name])
        results['dimensions'] = self.analyze_image_dimensions([split_name])
        results['annotations'] = self.analyze_annotations([split_name])
        results['file_sizes'] = self.analyze_file_sizes([split_name])
        results['complexity'] = self.analyze_complexity_distribution([split_name])

        return results

    def create_visualizations(self, splits: Optional[List[str]] = None):
        """Create visualization plots for specified splits"""
        print("\n=== CREATING VISUALIZATIONS ===")

        if splits is None:
            splits = list(self.coco_objects.keys())

        plt.style.use('seaborn-v0_8')

        # Determine number of subplots needed
        num_plots = 9
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Dataset Analysis Visualizations - Splits: {", ".join(splits)}', fontsize=16, fontweight='bold')

        # Color palette for different splits
        colors = plt.cm.Set1(np.linspace(0, 1, len(splits)))

        # 1. Category distribution
        if 'categories' in self.analysis_results:
            categories_data = self.analysis_results['categories']

            # Get all unique categories across splits
            all_categories = set()
            for split_name in splits:
                if split_name in categories_data:
                    all_categories.update(categories_data[split_name].keys())

            all_categories = sorted(list(all_categories))

            if all_categories:
                x = np.arange(len(all_categories))
                width = 0.8 / len(splits)

                for i, split_name in enumerate(splits):
                    if split_name in categories_data:
                        counts = [categories_data[split_name].get(cat, {}).get('images', 0) for cat in all_categories]
                        axes[0, 0].bar(x + i * width, counts, width, label=split_name, alpha=0.8, color=colors[i])

                axes[0, 0].set_xlabel('Categories')
                axes[0, 0].set_ylabel('Number of Images')
                axes[0, 0].set_title('Category Distribution')
                axes[0, 0].set_xticks(x + width * (len(splits) - 1) / 2)
                axes[0, 0].set_xticklabels(all_categories, rotation=45, ha='right')
                axes[0, 0].legend()

        # 2. Image dimensions
        if 'dimensions' in self.analysis_results:
            dims_data = self.analysis_results['dimensions']

            for i, split_name in enumerate(splits):
                if split_name in dims_data:
                    widths = dims_data[split_name]['raw_data']['widths']
                    heights = dims_data[split_name]['raw_data']['heights']

                    axes[0, 1].scatter(widths, heights, alpha=0.6, s=1, label=split_name, color=colors[i])

            axes[0, 1].set_xlabel('Width (pixels)')
            axes[0, 1].set_ylabel('Height (pixels)')
            axes[0, 1].set_title('Image Dimensions Distribution')
            if len(splits) > 1:
                axes[0, 1].legend()

        # 3. Aspect ratios
        if 'dimensions' in self.analysis_results:
            dims_data = self.analysis_results['dimensions']

            for i, split_name in enumerate(splits):
                if split_name in dims_data:
                    ratios = dims_data[split_name]['raw_data']['aspect_ratios']

                    axes[0, 2].hist(ratios, bins=50, alpha=0.7, label=split_name, color=colors[i])

            axes[0, 2].set_xlabel('Aspect Ratio')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Aspect Ratio Distribution')
            if len(splits) > 1:
                axes[0, 2].legend()

        # 4. Bounding box sizes
        if 'annotations' in self.analysis_results:
            ann_data = self.analysis_results['annotations']

            for i, split_name in enumerate(splits):
                if split_name in ann_data:
                    bbox_widths = ann_data[split_name]['raw_data']['bbox_widths']
                    bbox_heights = ann_data[split_name]['raw_data']['bbox_heights']

                    axes[1, 0].scatter(bbox_widths, bbox_heights, alpha=0.6, s=1, label=split_name, color=colors[i])

            axes[1, 0].set_xlabel('BBox Width (pixels)')
            axes[1, 0].set_ylabel('BBox Height (pixels)')
            axes[1, 0].set_title('Bounding Box Sizes')
            if len(splits) > 1:
                axes[1, 0].legend()

        # 5. Annotations per image
        if 'annotations' in self.analysis_results:
            ann_data = self.analysis_results['annotations']

            for i, split_name in enumerate(splits):
                if split_name in ann_data:
                    ann_per_img = ann_data[split_name]['raw_data']['annotations_per_image']

                    if ann_per_img:
                        max_ann = max(ann_per_img)
                        axes[1, 1].hist(ann_per_img, bins=range(0, max_ann + 2), alpha=0.7,
                                        label=split_name, color=colors[i])

            axes[1, 1].set_xlabel('Annotations per Image')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Annotations per Image Distribution')
            if len(splits) > 1:
                axes[1, 1].legend()

        # 6. File sizes
        if 'file_sizes' in self.analysis_results:
            file_data = self.analysis_results['file_sizes']

            for i, split_name in enumerate(splits):
                if split_name in file_data:
                    sizes = [size / (1024 * 1024) for size in file_data[split_name]['raw_data']]

                    axes[1, 2].hist(sizes, bins=50, alpha=0.7, label=split_name, color=colors[i])

            axes[1, 2].set_xlabel('File Size (MB)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('File Size Distribution')
            if len(splits) > 1:
                axes[1, 2].legend()

        # 7. Complexity distribution
        if 'complexity' in self.analysis_results:
            comp_data = self.analysis_results['complexity']

            # Get all unique object counts across splits
            all_object_counts = set()
            for split_name in splits:
                if split_name in comp_data:
                    all_object_counts.update(comp_data[split_name]['distribution'].keys())

            all_object_counts = sorted(list(all_object_counts))

            if all_object_counts:
                x = np.arange(len(all_object_counts))
                width = 0.8 / len(splits)

                for i, split_name in enumerate(splits):
                    if split_name in comp_data:
                        counts = [comp_data[split_name]['distribution'].get(obj_count, 0) for obj_count in
                                  all_object_counts]
                        axes[2, 0].bar(x + i * width, counts, width, label=split_name, alpha=0.7, color=colors[i])

                axes[2, 0].set_xlabel('Number of Objects')
                axes[2, 0].set_ylabel('Number of Images')
                axes[2, 0].set_title('Scene Complexity Distribution')
                axes[2, 0].set_xticks(x + width * (len(splits) - 1) / 2)
                axes[2, 0].set_xticklabels(all_object_counts)
                if len(splits) > 1:
                    axes[2, 0].legend()

        # 8. Category annotation counts
        if 'categories' in self.analysis_results:
            categories_data = self.analysis_results['categories']

            # Get all unique categories across splits
            all_categories = set()
            for split_name in splits:
                if split_name in categories_data:
                    all_categories.update(categories_data[split_name].keys())

            all_categories = sorted(list(all_categories))

            if all_categories:
                x = np.arange(len(all_categories))
                width = 0.8 / len(splits)

                for i, split_name in enumerate(splits):
                    if split_name in categories_data:
                        counts = [categories_data[split_name].get(cat, {}).get('annotations', 0) for cat in
                                  all_categories]
                        axes[2, 1].bar(x + i * width, counts, width, label=split_name, alpha=0.7, color=colors[i])

                axes[2, 1].set_xlabel('Categories')
                axes[2, 1].set_ylabel('Number of Annotations')
                axes[2, 1].set_title('Annotations per Category')
                axes[2, 1].set_xticks(x + width * (len(splits) - 1) / 2)
                axes[2, 1].set_xticklabels(all_categories, rotation=45, ha='right')
                if len(splits) > 1:
                    axes[2, 1].legend()

        # 9. Bounding box areas
        if 'annotations' in self.analysis_results:
            ann_data = self.analysis_results['annotations']

            for i, split_name in enumerate(splits):
                if split_name in ann_data:
                    bbox_areas = ann_data[split_name]['raw_data']['bbox_areas']

                    if bbox_areas:
                        log_areas = np.log10([area for area in bbox_areas if area > 0])
                        axes[2, 2].hist(log_areas, bins=50, alpha=0.7, label=split_name, color=colors[i])

            axes[2, 2].set_xlabel('Log10(BBox Area)')
            axes[2, 2].set_ylabel('Frequency')
            axes[2, 2].set_title('Bounding Box Area Distribution')
            if len(splits) > 1:
                axes[2, 2].legend()

        plt.tight_layout()

        # Create filename based on splits
        splits_str = "_".join(splits)
        filename = f'dataset_analysis_{splits_str}.png'
        plt.savefig(str(self.dataset_path / filename), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {self.dataset_path / filename}")

    def generate_detailed_readme(self):
        """Generate comprehensive README based on analysis results"""
        print("\n=== GENERATING DETAILED README ===")

        # Import datetime for timestamp
        from datetime import datetime

        basic_stats = self.analysis_results.get('basic_stats', {})
        categories = self.analysis_results.get('categories', {})
        dimensions = self.analysis_results.get('dimensions', {})
        annotations = self.analysis_results.get('annotations', {})
        file_sizes = self.analysis_results.get('file_sizes', {})
        complexity = self.analysis_results.get('complexity', {})

        # Get available splits for dynamic content
        available_splits = list(self.available_splits.keys())

        readme_content = f"""# Dataset Analysis Report

    This document contains a comprehensive analysis of the COCO-format dataset.

    ## 📊 Dataset Overview

    ### Available Splits
    """

        # Dynamic split information
        for split_name in available_splits:
            split_info = self.available_splits[split_name]
            readme_content += f"""- **{split_name.upper()}**: {split_info['images']:,} images, {split_info['annotations']:,} annotations, {split_info['categories']} categories
    """

        readme_content += f"""
    ### Dataset Statistics
    - **Total Images**: {basic_stats.get('total', {}).get('images', 0):,}
    - **Total Annotations**: {basic_stats.get('total', {}).get('annotations', 0):,}
    - **Total Categories**: {basic_stats.get('total', {}).get('categories', 0)}
    - **Dataset Path**: `{self.dataset_path}`

    """

        # Add split distribution if multiple splits exist
        if len(available_splits) > 1:
            readme_content += "### Split Distribution\n"
            total_images = basic_stats.get('total', {}).get('images', 1)
            total_annotations = basic_stats.get('total', {}).get('annotations', 1)

            for split_name in available_splits:
                split_stats = basic_stats.get(split_name, {})
                img_pct = (split_stats.get('images', 0) / total_images) * 100
                ann_pct = (split_stats.get('annotations', 0) / total_annotations) * 100
                readme_content += f"- **{split_name.upper()}**: {img_pct:.1f}% images, {ann_pct:.1f}% annotations\n"

            readme_content += "\n"

        # Category Analysis
        readme_content += "## 🏷️ Category Analysis\n\n"

        for split_name in available_splits:
            if split_name in categories:
                readme_content += f"### {split_name.upper()} Split\n\n"
                readme_content += "| Category | Images | Annotations | Avg per Image |\n"
                readme_content += "|----------|--------|-------------|---------------|\n"

                split_categories = categories[split_name]
                # Sort by number of images (descending)
                sorted_cats = sorted(split_categories.items(),
                                     key=lambda x: x[1]['images'], reverse=True)

                for cat_name, cat_data in sorted_cats:
                    readme_content += f"| {cat_name.title()} | {cat_data['images']:,} | {cat_data['annotations']:,} | {cat_data['avg_annotations_per_image']:.2f} |\n"

                readme_content += "\n"

        # Image Dimensions Analysis
        readme_content += "## 📐 Image Dimensions Analysis\n\n"

        for split_name in available_splits:
            if split_name in dimensions:
                split_dims = dimensions[split_name]
                readme_content += f"### {split_name.upper()} Split\n\n"

                readme_content += "| Dimension | Min | Max | Mean | Median | Std Dev |\n"
                readme_content += "|-----------|-----|-----|------|--------|---------|\n"
                readme_content += f"| Width (px) | {split_dims['width']['min']:,} | {split_dims['width']['max']:,} | {split_dims['width']['mean']:.0f} | {split_dims['width']['median']:.0f} | {split_dims['width']['std']:.0f} |\n"
                readme_content += f"| Height (px) | {split_dims['height']['min']:,} | {split_dims['height']['max']:,} | {split_dims['height']['mean']:.0f} | {split_dims['height']['median']:.0f} | {split_dims['height']['std']:.0f} |\n"
                readme_content += f"| Aspect Ratio | {split_dims['aspect_ratio']['min']:.2f} | {split_dims['aspect_ratio']['max']:.2f} | {split_dims['aspect_ratio']['mean']:.2f} | {split_dims['aspect_ratio']['median']:.2f} | {split_dims['aspect_ratio']['std']:.2f} |\n"
                readme_content += f"| Area (px²) | {split_dims['area']['min']:,} | {split_dims['area']['max']:,} | {split_dims['area']['mean']:,.0f} | {split_dims['area']['median']:,.0f} | {split_dims['area']['std']:,.0f} |\n"
                readme_content += "\n"

        # Annotation Analysis
        readme_content += "## 🔍 Annotation Analysis\n\n"

        for split_name in available_splits:
            if split_name in annotations:
                split_ann = annotations[split_name]
                readme_content += f"### {split_name.upper()} Split\n\n"

                readme_content += "#### Bounding Box Statistics\n"
                readme_content += "| Metric | Min | Max | Mean | Median | Std Dev |\n"
                readme_content += "|--------|-----|-----|------|--------|---------|\n"
                readme_content += f"| BBox Width | {split_ann['bbox_width']['min']:.0f} | {split_ann['bbox_width']['max']:.0f} | {split_ann['bbox_width']['mean']:.0f} | {split_ann['bbox_width']['median']:.0f} | {split_ann['bbox_width']['std']:.0f} |\n"
                readme_content += f"| BBox Height | {split_ann['bbox_height']['min']:.0f} | {split_ann['bbox_height']['max']:.0f} | {split_ann['bbox_height']['mean']:.0f} | {split_ann['bbox_height']['median']:.0f} | {split_ann['bbox_height']['std']:.0f} |\n"
                readme_content += f"| BBox Area | {split_ann['bbox_area']['min']:.0f} | {split_ann['bbox_area']['max']:.0f} | {split_ann['bbox_area']['mean']:.0f} | {split_ann['bbox_area']['median']:.0f} | {split_ann['bbox_area']['std']:.0f} |\n"
                readme_content += f"| BBox Aspect Ratio | {split_ann['bbox_aspect_ratio']['min']:.2f} | {split_ann['bbox_aspect_ratio']['max']:.2f} | {split_ann['bbox_aspect_ratio']['mean']:.2f} | {split_ann['bbox_aspect_ratio']['median']:.2f} | {split_ann['bbox_aspect_ratio']['std']:.2f} |\n"
                readme_content += f"| Annotations per Image | {split_ann['annotations_per_image']['min']:.0f} | {split_ann['annotations_per_image']['max']:.0f} | {split_ann['annotations_per_image']['mean']:.2f} | {split_ann['annotations_per_image']['median']:.2f} | {split_ann['annotations_per_image']['std']:.2f} |\n"
                readme_content += "\n"

        # File Size Analysis
        readme_content += "## 💾 Storage Analysis\n\n"

        total_size_gb = 0
        total_files = 0

        for split_name in available_splits:
            if split_name in file_sizes:
                split_files = file_sizes[split_name]
                total_size_gb += split_files['total_size_gb']
                total_files += split_files['file_count']

                readme_content += f"### {split_name.upper()} Split\n"
                readme_content += f"- **Total Size**: {split_files['total_size_gb']:.2f} GB ({split_files['total_size_mb']:.0f} MB)\n"
                readme_content += f"- **File Count**: {split_files['file_count']:,}\n"
                readme_content += f"- **Average File Size**: {split_files['avg_file_size_mb']:.2f} MB\n"
                readme_content += f"- **Size Range**: {split_files['min_file_size_bytes'] / 1024:.0f} KB - {split_files['max_file_size_bytes'] / (1024 * 1024):.2f} MB\n\n"

        if total_size_gb > 0:
            readme_content += f"### Total Dataset Storage\n"
            readme_content += f"- **Total Size**: {total_size_gb:.2f} GB\n"
            readme_content += f"- **Total Files**: {total_files:,}\n\n"

        # Scene Complexity Analysis
        readme_content += "## 🎭 Scene Complexity Analysis\n\n"

        for split_name in available_splits:
            if split_name in complexity:
                split_comp = complexity[split_name]
                readme_content += f"### {split_name.upper()} Split\n\n"

                readme_content += "#### Complexity Distribution\n"
                readme_content += f"- **Simple Scenes** (0-1 objects): {split_comp['simple_scenes']:,} images ({split_comp['simple_percentage']:.1f}%)\n"
                readme_content += f"- **Moderate Scenes** (2-3 objects): {split_comp['moderate_scenes']:,} images ({split_comp['moderate_percentage']:.1f}%)\n"
                readme_content += f"- **Complex Scenes** (4+ objects): {split_comp['complex_scenes']:,} images ({split_comp['complex_percentage']:.1f}%)\n"
                readme_content += f"- **Average Objects per Image**: {split_comp['avg_objects_per_image']:.2f}\n\n"

                # Detailed distribution
                if 'distribution' in split_comp:
                    readme_content += "#### Detailed Object Count Distribution\n"
                    sorted_dist = sorted(split_comp['distribution'].items())
                    for objects, count in sorted_dist[:10]:  # Show top 10
                        total_images = sum(split_comp['distribution'].values())
                        percentage = (count / total_images) * 100
                        readme_content += f"- **{objects} objects**: {count:,} images ({percentage:.1f}%)\n"
                    readme_content += "\n"

        # Visualizations section
        readme_content += "## 📊 Visualizations\n\n"
        readme_content += "The analysis includes comprehensive visualizations showing:\n\n"
        readme_content += "1. **Category Distribution** - Image and annotation counts per category\n"
        readme_content += "2. **Image Dimensions** - Width vs height scatter plot\n"
        readme_content += "3. **Aspect Ratios** - Distribution of image aspect ratios\n"
        readme_content += "4. **Bounding Box Sizes** - Distribution of annotation dimensions\n"
        readme_content += "5. **Annotations per Image** - Scene complexity histogram\n"
        readme_content += "6. **File Sizes** - Storage requirements distribution\n"
        readme_content += "7. **Scene Complexity** - Object count per image\n"
        readme_content += "8. **Category Annotations** - Detailed annotation counts\n"
        readme_content += "9. **Bounding Box Areas** - Size distribution of objects\n\n"

        # Dataset structure
        readme_content += "## 📁 Dataset Structure\n\n"
        readme_content += "```\n"
        readme_content += f"{self.dataset_path.name}/\n"
        readme_content += "├── annotations/\n"

        for split_name in available_splits:
            split_info = self.available_splits[split_name]
            readme_content += f"│   ├── {split_info['annotation_file'].name}\n"

        for split_name in available_splits:
            split_info = self.available_splits[split_name]
            if split_info['image_dir']:
                readme_content += f"├── {split_info['image_dir'].name}/\n"

        readme_content += "├── dataset_analysis_*.png\n"
        readme_content += "├── analysis_results_*.json\n"
        readme_content += "└── README.md\n"
        readme_content += "```\n\n"

        # Technical recommendations
        readme_content += "## 🚀 Training Recommendations\n\n"
        readme_content += "### Data Augmentation\n"
        readme_content += "- **Horizontal flips**: Safe for most object categories\n"
        readme_content += "- **Scaling**: Consider object size distribution\n"
        readme_content += "- **Rotation**: Use with caution to maintain object appearance\n"
        readme_content += "- **Color adjustments**: Brightness, contrast, saturation\n"
        readme_content += "- **Cropping**: Respect aspect ratios and object boundaries\n\n"

        readme_content += "### Model Configuration\n"
        readme_content += "- **Input size**: 640x640 or 832x832 for better accuracy\n"
        readme_content += "- **Batch size**: 8-16 depending on GPU memory\n"
        readme_content += "- **Learning rate**: Start with 0.001 and adjust based on validation\n\n"

        readme_content += "### Evaluation Metrics\n"
        readme_content += "- **mAP@0.5**: Primary metric for object detection\n"
        readme_content += "- **mAP@0.5:0.95**: Comprehensive evaluation across IoU thresholds\n"
        readme_content += "- **Per-category metrics**: Monitor individual category performance\n\n"

        # Footer
        readme_content += "---\n\n"
        readme_content += "*This analysis was automatically generated by the Dataset Analyzer*\n\n"
        readme_content += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        # Save README
        readme_path = self.dataset_path / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"Detailed README saved to: {readme_path}")
        return readme_content

    def run_complete_analysis(self, splits: Optional[List[str]] = None):
        """Run complete analysis pipeline for specified splits"""
        if splits is None:
            splits = list(self.coco_objects.keys())

        print(f"Starting comprehensive dataset analysis for splits: {', '.join(splits)}...")

        # Run all analyses
        self.analyze_basic_stats()
        self.analyze_categories(splits)
        self.analyze_image_dimensions(splits)
        self.analyze_annotations(splits)
        self.analyze_file_sizes(splits)
        self.analyze_complexity_distribution(splits)

        # Create visualizations
        self.create_visualizations(splits)

        # Generate README - ADD THIS LINE
        self.generate_detailed_readme()

        # Save analysis results
        splits_str = "_".join(splits)
        results_filename = f'analysis_results_{splits_str}.json'

        with open(self.dataset_path / results_filename, 'w') as f:
            json_results = json.loads(json.dumps(self.analysis_results, default=str))
            json.dump(json_results, f, indent=2)

        print(f"Analysis results saved to: {self.dataset_path / results_filename}")
        return self.analysis_results

    def run_train_analysis(self):
        """Run analysis specifically for training split"""
        if 'train' in self.coco_objects:
            return self.run_complete_analysis(['train'])
        else:
            print("Warning: 'train' split not found in dataset")
            return None

    def run_val_analysis(self):
        """Run analysis specifically for validation split"""
        if 'val' in self.coco_objects:
            return self.run_complete_analysis(['val'])
        else:
            print("Warning: 'val' split not found in dataset")
            return None

    def run_test_analysis(self):
        """Run analysis specifically for test split"""
        if 'test' in self.coco_objects:
            return self.run_complete_analysis(['test'])
        else:
            print("Warning: 'test' split not found in dataset")
            return None

    def get_available_splits(self):
        """Get list of available splits in the dataset"""
        return list(self.available_splits.keys())

    def print_dataset_summary(self):
        """Print a comprehensive summary of the dataset"""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"Dataset Path: {self.dataset_path}")
        print(f"Available Splits: {', '.join(self.available_splits.keys())}")

        print("\nSplit Details:")
        for split_name, info in self.available_splits.items():
            print(f"  {split_name.upper()}:")
            print(f"    Images: {info['images']:,}")
            print(f"    Annotations: {info['annotations']:,}")
            print(f"    Categories: {info['categories']}")
            print(f"    Annotation File: {info['annotation_file'].name}")
            print(f"    Image Directory: {info['image_dir'] if info['image_dir'] else 'Not found'}")

        # Total statistics
        total_images = sum(info['images'] for info in self.available_splits.values())
        total_annotations = sum(info['annotations'] for info in self.available_splits.values())

        print(f"\nTotal Dataset Statistics:")
        print(f"  Total Images: {total_images:,}")
        print(f"  Total Annotations: {total_annotations:,}")
        print(
            f"  Average Annotations per Image: {total_annotations / total_images:.2f}" if total_images > 0 else "  Average Annotations per Image: 0")

        # Category information
        all_categories = set()
        for coco_obj in self.coco_objects.values():
            for cat_id in coco_obj.getCatIds():
                cat_info = coco_obj.loadCats([cat_id])[0]
                all_categories.add(cat_info['name'])

        print(f"  Unique Categories: {len(all_categories)}")
        if len(all_categories) <= 20:  # Only show if not too many
            print(f"  Categories: {', '.join(sorted(all_categories))}")

        print("=" * 60)


# Example usage and utility functions
def analyze_dataset(dataset_path: str, splits: Optional[List[str]] = None):
    """Convenience function to analyze a dataset"""
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.print_dataset_summary()

    if splits is None:
        # Analyze all available splits
        results = analyzer.run_complete_analysis()
    else:
        # Analyze specific splits
        results = analyzer.run_complete_analysis(splits)

    return analyzer, results


def analyze_train_val_separately(dataset_path: str):
    """Analyze train and validation splits separately"""
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.print_dataset_summary()

    results = {}

    # Analyze train split if available
    if 'train' in analyzer.get_available_splits():
        print("\n" + "=" * 40)
        print("ANALYZING TRAIN SPLIT SEPARATELY")
        print("=" * 40)
        train_results = analyzer.run_train_analysis()
        results['train'] = train_results

    # Analyze validation split if available
    if 'val' in analyzer.get_available_splits():
        print("\n" + "=" * 40)
        print("ANALYZING VALIDATION SPLIT SEPARATELY")
        print("=" * 40)
        val_results = analyzer.run_val_analysis()
        results['val'] = val_results

    return analyzer, results


def analyze_single_split_dataset(dataset_path: str, split_name: str = None):
    """Analyze a dataset with a single split (e.g., test set)"""
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.print_dataset_summary()

    available_splits = analyzer.get_available_splits()

    if split_name is None:
        # Use the first available split
        if available_splits:
            split_name = available_splits[0]
        else:
            raise ValueError("No splits found in dataset")

    if split_name not in available_splits:
        raise ValueError(f"Split '{split_name}' not found. Available splits: {available_splits}")

    print(f"\n" + "=" * 40)
    print(f"ANALYZING {split_name.upper()} SPLIT")
    print("=" * 40)

    results = analyzer.run_complete_analysis([split_name])
    return analyzer, results


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Analyze COCO-format datasets')
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('--splits', nargs='*', help='Specific splits to analyze (e.g., train val test)')
    parser.add_argument('--separate', action='store_true', help='Analyze train/val splits separately')
    parser.add_argument('--single', help='Analyze a single split (specify split name)')

    args = parser.parse_args()

    try:
        if args.separate:
            analyzer, results = analyze_train_val_separately(args.dataset_path)
        elif args.single:
            analyzer, results = analyze_single_split_dataset(args.dataset_path, args.single)
        else:
            analyzer, results = analyze_dataset(args.dataset_path, args.splits)

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()