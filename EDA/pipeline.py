from pathlib import Path
from config import create_config
from loader import create_coco_dataframe
from metadata import create_merged_df
from visualize_examples import visualize_examples
from visualize_stats import visualize_stats


def run_full_pipeline(config):
    """Run the complete EDA pipeline."""
    print("Starting EDA pipeline...")

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print("Loading COCO dataset...")
    df_from_coco = create_coco_dataframe(config)
    print(f"Loaded {len(df_from_coco)} annotations")

    # 2. Create metadata
    print("Creating metadata...")
    df_merged = create_merged_df(df_from_coco)

    # 3. Create visualizations
    print("Creating example visualizations...")
    visualize_examples(df_merged, config)

    print("Creating statistical visualizations...")
    visualize_stats(df_merged, config)

    print(f"Pipeline completed! Results saved to {config.output_dir}")
    return df_merged


if __name__ == "__main__":
    # For simple script execution without command line args
    config = create_config()

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading dataset...")
    df_from_coco = create_coco_dataframe(config)
    df_merged = create_merged_df(df_from_coco)
    print(f"Loaded {len(df_merged)} annotations from {df_merged['image_id'].nunique()} images")

    visualize_examples(df_merged, config)

    visualize_stats(df_merged, config)

    print(f"\nPipeline completed! Check results in {config.output_dir}")
