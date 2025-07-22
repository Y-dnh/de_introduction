from pycocotools.coco import COCO
import pandas as pd
import os
import re

class COCODatasetLoader:
    """
    Loads COCO annotations and merges with image metadata into a single pandas DataFrame.
    """

    def __init__(self, config):
        self.dataset_dir = config.dataset_dir
        self.df = None  # Placeholder for the final DataFrame

    def load_df_dataset(self):
        """
        Parse all COCO JSON files in the `annotations` directory, merge
        annotation-level and image-level info, and store in self.df.
        """

        annotations_dir = os.path.join(self.dataset_dir, "annotations")
        json_files = self._parse_annotations(annotations_dir)  # Get list of JSON files (e.g., instances_train.json, instances_val.json)
        dfs = []

        for fname in json_files:
            # Infer dataset split from filename (train/val/test)
            split_name = self._infer_split(fname)
            ann_path = os.path.join(annotations_dir, fname)
            coco = COCO(str(ann_path))

            df_anns = pd.DataFrame(coco.loadAnns(coco.getAnnIds()))  # Convert annotation dicts to DataFrame
            df_imgs = pd.DataFrame(coco.loadImgs(coco.getImgIds()))  # Convert image metadata dicts to DataFrame

            df = df_anns.merge(
                df_imgs,
                left_on="image_id",
                right_on="id",
                suffixes=("", "_img")
            )
            df["split"] = split_name

            desired = ["image_id", "file_name", "width",
                       "height", "category_id", "bbox",
                       "area", "iscrowd", "segmentation", "split",
            ]
            keep = [c for c in desired if c in df.columns]  # Only keep columns that exist (handles Pascal VOC without some fields)
            dfs.append(df[keep])

        self.df = pd.concat(dfs, ignore_index=True)

    @staticmethod
    def _parse_annotations(annotations_dir: str) -> list[str]:
        if not os.path.isdir(annotations_dir):
            raise FileNotFoundError(f"{annotations_dir} does not exist")
        json_files = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No .json files in {annotations_dir}")
        return json_files

    @staticmethod
    def _infer_split(filename: str) -> str:
        fname = filename.lower()
        if re.search(r"train", fname):
            return "train"
        if re.search(r"val", fname):
            return "val"
        if re.search(r"test", fname):
            return "test"
        return "unknown"


def create_coco_dataframe(config) -> pd.DataFrame:
    loader = COCODatasetLoader(config)
    loader.load_df_dataset()
    return loader.df


if __name__ == "__main__":
    # Example usage
    from config import create_config
    # Show all columns when printing
    pd.set_option('display.max_columns', None)
    # Load DataFrame and display head
    df = create_coco_dataframe(create_config())
    print(df.head())