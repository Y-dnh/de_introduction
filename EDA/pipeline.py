import argparse
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


def run_examples_only(config):
    """Run only example visualizations."""
    print("Running examples visualization...")
    df_from_coco = create_coco_dataframe(config)
    df_merged = create_merged_df(df_from_coco)
    visualize_examples(df_merged, config)
    print("Examples visualization completed!")


def run_stats_only(config):
    """Run only statistical visualizations."""
    print("Running statistical analysis...")
    df_from_coco = create_coco_dataframe(config)
    df_merged = create_merged_df(df_from_coco)
    visualize_stats(df_merged, config)
    print("Statistical analysis completed!")


def main():
    """Main pipeline coordinator with command line interface."""
    parser = argparse.ArgumentParser(description='COCO Dataset EDA Pipeline')
    parser.add_argument('--mode', choices=['full', 'examples', 'stats'],
                        default='full', help='Pipeline mode to run')
    parser.add_argument('--config', type=str, help='Custom config file (optional)')

    args = parser.parse_args()

    # Load configuration
    config = create_config()

    # Run selected pipeline
    if args.mode == 'full':
        run_full_pipeline(config)
    elif args.mode == 'examples':
        run_examples_only(config)
    elif args.mode == 'stats':
        run_stats_only(config)


if __name__ == "__main__":
    # For simple script execution without command line args
    config = create_config()

    # Choose what to run (modify as needed):
    RUN_EXAMPLES = True
    RUN_STATS = True

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data once
    print("Loading dataset...")
    df_from_coco = create_coco_dataframe(config)
    df_merged = create_merged_df(df_from_coco)
    print(f"Loaded {len(df_merged)} annotations from {df_merged['image_id'].nunique()} images")

    # Run selected components
    if RUN_EXAMPLES:
        print("\nCreating example visualizations...")
        visualize_examples(df_merged, config)

    if RUN_STATS:
        print("\nCreating statistical visualizations...")
        visualize_stats(df_merged, config)

    print(f"\nPipeline completed! Check results in {config.output_dir}")

    # For command line usage, uncomment:
    # main()