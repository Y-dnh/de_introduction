import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pandas as pd

def _decode_segmentation(segmentation, img_height, img_width):
    """Convert COCO segmentation format to binary mask."""
    if not segmentation or segmentation == []:
        return None

    # Handle RLE format (compressed)
    if isinstance(segmentation, dict):
        return None  # Skip RLE for simplicity

    # Handle polygon format
    if isinstance(segmentation, list) and len(segmentation) > 0:
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for poly in segmentation:
            if len(poly) >= 6:  # At least 3 points (x,y pairs)
                poly_array = np.array(poly).reshape(-1, 2)
                # Simple polygon fill using matplotlib Path
                from matplotlib.path import Path as MplPath
                path = MplPath(poly_array)
                y, x = np.mgrid[:img_height, :img_width]
                points = np.column_stack((x.ravel(), y.ravel()))
                mask_flat = path.contains_points(points)
                mask += mask_flat.reshape(img_height, img_width)
        return mask > 0

    return None


def _filter_by_bbox_ratio(df, min_pct, max_pct):
    """Filter DataFrame by bbox_to_img_ratio."""
    return df[
        (df['bbox_to_img_ratio'] >= min_pct) &
        (df['bbox_to_img_ratio'] <= max_pct)
        ]


def _create_example_visualization(df_sample, dataset_dir, output_dir, config):
    """Create separate visualization images for each example."""
    n_examples = min(config.visualization.n_examples, len(df_sample))
    unique_ids = df_sample['image_id'].drop_duplicates().sample(n=n_examples, random_state=42)
    sample_df = df_sample[df_sample['image_id'].isin(unique_ids)]

    # Group by image to handle multiple objects per image
    grouped = sample_df.groupby('image_id')

    colors = config.visualization.color_palette

    for idx, (image_id, group) in enumerate(grouped):
        if idx >= n_examples:
            break

        fig, ax = plt.subplots(figsize=config.visualization.figure_size, dpi=config.visualization.dpi)

        # Load image
        img_path = os.path.join(dataset_dir, group.iloc[0]['file_name'])
        if not os.path.exists(img_path):
            # Try different common subdirectories
            for subdir in ['images', 'train', 'val', 'test']:
                alt_path = os.path.join(dataset_dir, subdir, group.iloc[0]['file_name'])
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break

        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        except:
            # Create placeholder if image not found
            img_height, img_width = group.iloc[0]['height'], group.iloc[0]['width']
            placeholder = np.ones((img_height, img_width, 3)) * 0.5
            ax.imshow(placeholder)
            ax.text(img_width // 2, img_height // 2, 'Image\nNot Found',
                    ha='center', va='center', fontsize=12, color='red')

        # Draw annotations
        for ann_idx, (_, row) in enumerate(group.iterrows()):
            color = colors[ann_idx % len(colors)]

            # Bounding box
            x, y, w, h = row['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Segmentation
            if 'segmentation' in row and row['segmentation']:
                mask = _decode_segmentation(row['segmentation'],
                                            int(row['height']), int(row['width']))
                if mask is not None:
                    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
                    hex_color = color.lstrip('#')
                    rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
                    colored_mask[mask] = (*rgb, 0.3)
                    ax.imshow(colored_mask)

            # Bbox area
            bbox_area = int(row['bbox_area'])
            bbox_pct = row['bbox_to_img_ratio'] * 100

            # Segmentation area
            if not pd.isna(row.get('seg_area', np.nan)):
                seg_area = int(row['seg_area'])
                seg_pct = row['seg_to_img_ratio'] * 100
            else:
                seg_area = None
                seg_pct = None

            lines = [
                f"Box: {bbox_area}px ({bbox_pct:.1f}%)"
            ]
            if seg_area is not None:
                lines.append(f"Seg: {seg_area}px ({seg_pct:.1f}%)")
            info_text = "\n".join(lines)

            ax.text(
                x, y - 2,
                info_text,
                va='bottom', ha='left',
                fontsize=7, color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )

        ax.set_title(f'Image {image_id} — {len(group)} objects')
        ax.axis('off')

        plt.tight_layout()

        # Save individual image
        output_path = os.path.join(output_dir, f"example_{idx}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=config.visualization.dpi)
        plt.close()



def visualize_examples(df, config):
    """Create example visualizations for different bbox threshold configurations."""
    output_dir = Path(config.output_dir)
    bbox_output_dir = output_dir / "bbox_thresholds"
    bbox_output_dir.mkdir(parents=True, exist_ok=True)

    # Handle both single values and lists for min_pct/max_pct
    min_pcts = config.bbox.min_pct if isinstance(config.bbox.min_pct, list) else [config.bbox.min_pct]
    max_pcts = config.bbox.max_pct if isinstance(config.bbox.max_pct, list) else [config.bbox.max_pct]

    if len(min_pcts) != len(max_pcts):
        raise ValueError("min_pct and max_pct lists must have the same length")

    for min_pct, max_pct in zip(min_pcts, max_pcts):
        # Create folder name
        folder_name = f"{min_pct}_{max_pct}"
        threshold_dir = bbox_output_dir / folder_name
        threshold_dir.mkdir(exist_ok=True)

        # Filter data by bbox ratio
        df_filtered = _filter_by_bbox_ratio(df, min_pct, max_pct)

        if df_filtered.empty:
            print(f"No data found for threshold {min_pct}-{max_pct}")
            continue

        _create_example_visualization(df_filtered, config.dataset_dir, threshold_dir, config)

        print(f"Created examples for threshold {min_pct}-{max_pct}: {len(df_filtered)} annotations")


if __name__ == "__main__":
    from config import create_config
    from loader import create_coco_dataframe
    from metadata import create_merged_df

    config = create_config()
    df_from_coco = create_coco_dataframe(config)
    df_merged = create_merged_df(df_from_coco)

    visualize_examples(df_merged, config)