from pycocotools.coco import COCO
import pandas as pd
import os
import re

class COCODatasetLoader:
    def __init__(self, config):
        self.annotations_dir = None
        self.dataset_anns = None
        self.dataset_dir = config.dataset_dir
        self.df = None

    def load_df_dataset(self):
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        json_files = self._parse_annotations()
        dfs = []

        for fname in json_files:
            split_name = self._infer_split(fname)
            ann_path = os.path.join(self.annotations_dir, fname)
            coco = COCO(ann_path)

            anns = coco.loadAnns(coco.getAnnIds())
            imgs = coco.loadImgs(coco.getImgIds())

            df_anns = pd.DataFrame(anns)
            df_imgs = pd.DataFrame(imgs)

            # merge annotations with image info
            df = df_anns.merge(df_imgs, left_on="image_id", right_on="id", suffixes=('', '_img'))
            df["split"] = split_name
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

    def _parse_annotations(self):
        if not os.path.isdir(self.annotations_dir):
            raise FileNotFoundError("{} does not exist".format(self.annotations_dir))

        json_files = [f for f in os.listdir(self.annotations_dir) if f.endswith(".json")]

        if not json_files:
            raise FileNotFoundError("No .json files in {}".format(self.annotations_dir))

        return json_files


    @staticmethod
    def _infer_split(filename):
        fname = filename.lower()
        if re.search(r"train", fname):
            return "train"
        elif re.search(r"val", fname):
            return "val"
        elif re.search(r"test", fname):
            return "test"
        return "unknown"


def create_coco_dataframe(config):
    coco_df = COCODatasetLoader(config)
    coco_df.load_df_dataset()
    return coco_df.df.drop("id", axis=1).drop("id_img", axis=1)  # image_id is repeated, id = row indexing + 1


if __name__ == "__main__":
    from config import create_config
    pd.set_option('display.max_columns', None)
    coco_df = COCODatasetLoader(create_config())
    coco_df.load_df_dataset()
    df = coco_df.df
    print(df.head())