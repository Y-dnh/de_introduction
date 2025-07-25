import matplotlib.pyplot as plt
from pathlib import Path


def _create_image_stats(df, config, output_dir):
    """Create histograms for image-level statistics."""
    fig, axes = plt.subplots(2, 2, figsize=config.visualization.figure_size, dpi=config.visualization.dpi)
    axes = axes.flatten()

    # Get unique images (remove duplicates from multiple annotations)
    df_images = df.drop_duplicates(subset=['image_id'])

    # Image width distribution
    axes[0].hist(df_images['width'], bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[0], alpha=0.7, edgecolor='black')
    axes[0].set_title('Image Width Distribution')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Image height distribution
    axes[1].hist(df_images['height'], bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[1], alpha=0.7, edgecolor='black')
    axes[1].set_title('Image Height Distribution')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    # Image aspect ratio distribution
    axes[2].hist(df_images['img_aspect_ratio'], bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[2], alpha=0.7, edgecolor='black')
    axes[2].set_title('Image Aspect Ratio Distribution')
    axes[2].set_xlabel('Aspect Ratio (width/height)')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)

    # Objects per image distribution
    objects_per_image = df.groupby('image_id').size()
    axes[3].hist(objects_per_image, bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[3], alpha=0.7, edgecolor='black')
    axes[3].set_title('Objects per Image Distribution')
    axes[3].set_xlabel('Number of Objects')
    axes[3].set_ylabel('Count')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "image_statistics.png", bbox_inches='tight', dpi=config.visualization.dpi)
    plt.close()


def _create_bbox_stats(df, config, output_dir):
    """Create histograms for bounding box statistics."""
    fig, axes = plt.subplots(2, 2, figsize=config.visualization.figure_size, dpi=config.visualization.dpi)
    axes = axes.flatten()

    # BBox area distribution
    axes[0].hist(df['bbox_area'], bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[0], alpha=0.7, edgecolor='black')
    axes[0].set_title('Bounding Box Area Distribution')
    axes[0].set_xlabel('Area (pixels²)')
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # BBox to image ratio distribution
    axes[1].hist(df['bbox_to_img_ratio'], bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[1], alpha=0.7, edgecolor='black')
    axes[1].set_title('BBox to Image Ratio Distribution')
    axes[1].set_xlabel('Ratio (bbox_area / img_area)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    # BBox width distribution
    bbox_widths = df['bbox'].apply(lambda x: x[2])
    axes[2].hist(bbox_widths, bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[2], alpha=0.7, edgecolor='black')
    axes[2].set_title('Bounding Box Width Distribution')
    axes[2].set_xlabel('Width (pixels)')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3)

    # BBox height distribution
    bbox_heights = df['bbox'].apply(lambda x: x[3])
    axes[3].hist(bbox_heights, bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[3], alpha=0.7, edgecolor='black')
    axes[3].set_title('Bounding Box Height Distribution')
    axes[3].set_xlabel('Height (pixels)')
    axes[3].set_ylabel('Count')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bbox_statistics.png", bbox_inches='tight', dpi=config.visualization.dpi)
    plt.close()


def _create_class_stats(df, config, output_dir):
    """Create histograms for class-specific statistics."""
    # Class distribution
    class_counts = df['category_id'].value_counts().sort_index()

    fig, axes = plt.subplots(2, 1, figsize=config.visualization.figure_size, dpi=config.visualization.dpi)

    # Overall class distribution
    class_names = [config.classes[i % len(config.classes)] for i in class_counts.index]
    bars = axes[0].bar(class_names, class_counts.values,
                       color=config.visualization.color_palette[:len(class_names)], alpha=0.7)
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(class_counts.values) * 0.01,
                     str(count), ha='center', va='bottom')

    # BBox size by class
    class_bbox_areas = []
    class_labels = []

    for cat_id in class_counts.index:
        class_data = df[df['category_id'] == cat_id]
        class_bbox_areas.extend(class_data['bbox_area'].tolist())
        class_name = config.classes[cat_id % len(config.classes)]
        class_labels.extend([class_name] * len(class_data))

    # Create boxplot for bbox areas by class
    unique_classes = [config.classes[i % len(config.classes)] for i in sorted(class_counts.index)]
    bbox_data_by_class = [df[df['category_id'] == cat_id]['bbox_area'].tolist()
                          for cat_id in sorted(class_counts.index)]

    bp = axes[1].boxplot(bbox_data_by_class, labels=unique_classes, patch_artist=True)
    for patch, color in zip(bp['boxes'], config.visualization.color_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_title('Bounding Box Area by Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('BBox Area (pixels²)')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "class_statistics.png", bbox_inches='tight', dpi=config.visualization.dpi)
    plt.close()


def _create_segmentation_stats(df, config, output_dir):
    """Create histograms for segmentation statistics (if available)."""
    if 'seg_area' not in df.columns:
        return

    # Filter out rows without segmentation
    df_with_seg = df.dropna(subset=['seg_area'])

    if df_with_seg.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=config.visualization.figure_size, dpi=config.visualization.dpi)

    # Segmentation to bbox ratio
    axes[0].hist(df_with_seg['seg_to_bbox_ratio'].dropna(), bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[0], alpha=0.7, edgecolor='black')
    axes[0].set_title('Segmentation to BBox Ratio')
    axes[0].set_xlabel('Ratio (seg_area / bbox_area)')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Segmentation to image ratio
    axes[1].hist(df_with_seg['seg_to_img_ratio'].dropna(), bins=config.visualization.histogram_bins,
                 color=config.visualization.color_palette[1], alpha=0.7, edgecolor='black')
    axes[1].set_title('Segmentation to Image Ratio')
    axes[1].set_xlabel('Ratio (seg_area / img_area)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "segmentation_statistics.png", bbox_inches='tight', dpi=config.visualization.dpi)
    plt.close()


def _create_split_stats(df, config, output_dir):
    """Create statistics by dataset split."""
    if 'split' not in df.columns:
        return

    split_counts = df['split'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=config.visualization.figure_size, dpi=config.visualization.dpi)

    # Split distribution
    bars = axes[0].bar(split_counts.index, split_counts.values,
                       color=config.visualization.color_palette[:len(split_counts)], alpha=0.7)
    axes[0].set_title('Dataset Split Distribution')
    axes[0].set_xlabel('Split')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # Add value labels
    for bar, count in zip(bars, split_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(split_counts.values) * 0.01,
                     str(count), ha='center', va='bottom')

    # BBox ratio distribution by split
    split_data = [df[df['split'] == split]['bbox_to_img_ratio'].tolist() for split in split_counts.index]
    bp = axes[1].boxplot(split_data, labels=split_counts.index, patch_artist=True)
    for patch, color in zip(bp['boxes'], config.visualization.color_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_title('BBox to Image Ratio by Split')
    axes[1].set_xlabel('Split')
    axes[1].set_ylabel('BBox to Image Ratio')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "split_statistics.png", bbox_inches='tight', dpi=config.visualization.dpi)
    plt.close()


def visualize_stats(df, config):
    """Create comprehensive statistical visualizations."""
    output_dir = Path(config.output_dir)
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("Creating image statistics...")
    _create_image_stats(df, config, stats_dir)

    print("Creating bounding box statistics...")
    _create_bbox_stats(df, config, stats_dir)

    print("Creating class statistics...")
    _create_class_stats(df, config, stats_dir)

    print("Creating segmentation statistics...")
    _create_segmentation_stats(df, config, stats_dir)

    print("Creating split statistics...")
    _create_split_stats(df, config, stats_dir)

    # Create summary statistics file
    _create_summary_stats(df, config, stats_dir)

    print(f"Statistics saved to {stats_dir}")


def _create_summary_stats(df, config, output_dir):
    """Create a text summary of key statistics."""
    with open(output_dir / "summary_stats.txt", "w") as f:
        f.write("Dataset Summary Statistics\n")
        f.write("=" * 30 + "\n\n")

        # Basic counts
        f.write(f"Total annotations: {len(df)}\n")
        f.write(f"Unique images: {df['image_id'].nunique()}\n")
        f.write(f"Unique classes: {df['category_id'].nunique()}\n\n")

        # Image statistics
        df_images = df.drop_duplicates(subset=['image_id'])
        f.write("Image Statistics:\n")
        f.write(f"  Average width: {df_images['width'].mean():.1f}\n")
        f.write(f"  Average height: {df_images['height'].mean():.1f}\n")
        f.write(f"  Average aspect ratio: {df_images['img_aspect_ratio'].mean():.2f}\n\n")

        # BBox statistics
        f.write("Bounding Box Statistics:\n")
        f.write(f"  Average bbox area: {df['bbox_area'].mean():.1f}\n")
        f.write(f"  Average bbox to image ratio: {df['bbox_to_img_ratio'].mean():.3f}\n")
        f.write(f"  Median bbox to image ratio: {df['bbox_to_img_ratio'].median():.3f}\n\n")

        # Class distribution
        f.write("Class Distribution:\n")
        class_counts = df['category_id'].value_counts().sort_index()
        for cat_id, count in class_counts.items():
            class_name = config.classes[cat_id % len(config.classes)]
            f.write(f"  {class_name}: {count}\n")

        # Split distribution (if available)
        if 'split' in df.columns:
            f.write("\nSplit Distribution:\n")
            split_counts = df['split'].value_counts()
            for split, count in split_counts.items():
                f.write(f"  {split}: {count}\n")


if __name__ == "__main__":
    from config import create_config
    from loader import create_coco_dataframe
    from metadata import create_merged_df

    config = create_config()
    df_from_coco = create_coco_dataframe(config)
    df_merged = create_merged_df(df_from_coco)

    visualize_stats(df_merged, config)