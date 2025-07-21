from pycocotools.coco import COCO
import pandas as pd
from config import config
import os

class COCODatasetLoader:
    def __init__(self):
        self.annotations_dir = None
        self.dataset_anns = None
        self.dataset_dir = config.dataset_dir
        self.coco = None
        self.df = None

    def load_dataset(self):
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        json_files = self._parse_annotations()
        dfs = []

        for fname in json_files:
            split_name = self._infer_split(fname)  # train/val/test/unknown
            ann_path = os.path.join(self.annotations_dir, fname)
            coco = COCO(ann_path)
            anns = coco.loadAnns(coco.getAnnIds())
            df = pd.DataFrame(anns)
            df["split"] = split_name
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

    def _parse_annotations(self):
        if not os.path.isdir(self.annotations_dir):
            print("Не знайдено директорії 'annotations'")
            return

        json_files = [f for f in os.listdir(self.annotations_dir) if f.endswith(".json")]

        if not json_files:
            raise("У 'annotations' немає .json файлів")

        return json_files

    def _coco_to_dataframe(self) -> pd.DataFrame:
        anns = self.coco.loadAnns(self.coco.getAnnIds(self.dataset_anns))
        return pd.DataFrame(anns)

    def _infer_split(self, filename):
        fname = filename.lower()
        if "train" in fname:
            return "train"
        elif "val" in fname:
            return "val"
        elif "test" in fname:
            return "test"
        else:
            return "unknown"


if __name__ == "__main__":
    loader = COCODatasetLoader()
    loader.load_dataset()
    print(loader.df.head())