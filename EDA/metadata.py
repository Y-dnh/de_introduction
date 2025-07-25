import numpy as np
import pandas as pd


def create_merged_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a COCO-style annotations DataFrame with additional derived metadata columns.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with these added or renamed columns:

        - img_area             : float
            Total pixel area of the image (width * height).
        - bbox_area            : float
            Area of each bounding box (w * h).
        - seg_area             : float
            If a segmentation mask exists, this is the mask area;
            otherwise equals bbox_area.
        - bbox_to_img_ratio    : float
            Fraction of image occupied by the bounding box (bbox_area / img_area).
        - seg_to_bbox_ratio    : float
            Fraction of the bounding box filled by the mask (seg_area / bbox_area).
        - seg_to_img_ratio     : float
            Fraction of image occupied by the mask (seg_area / img_area).
        - x_center             : float
            X-coordinate of the bounding box center (x + w/2).
        - y_center             : float
            Y-coordinate of the bounding box center (y + h/2).
        - img_aspect_ratio     : float
            Aspect ratio of the image (width / height).
    """

    df_merged = df.copy()

    # 1) Image area
    df_merged["img_area"] = df_merged["width"] * df_merged["height"]

    # 2) BBox area
    df_merged["bbox_area"] = df_merged["bbox"].apply(lambda b: b[2] * b[3])
    df_merged["bbox_to_img_ratio"] = df_merged["bbox_area"] / df_merged["img_area"]

    # 3) Always compute center coords and aspect ratio
    df_merged["x_center"] = df_merged["bbox"].apply(lambda b: b[0] + b[2] / 2)
    df_merged["y_center"] = df_merged["bbox"].apply(lambda b: b[1] + b[3] / 2)
    df_merged["img_aspect_ratio"] = df_merged["width"] / df_merged["height"]

    # 4) Only if there is a 'segmentation' column...
    if "segmentation" in df_merged.columns:
        # 4a) Rename original 'area' → 'seg_area' if present
        if "area" in df_merged.columns:
            df_merged = df_merged.rename(columns={"area": "seg_area"})
        else:
            df_merged["seg_area"] = np.nan

        # 4b) Keep seg_area only where a non‐empty mask exists
        #     Otherwise leave it NaN
        df_merged["seg_area"] = df_merged.apply(
            lambda row: row["seg_area"]
            if (bool(row.get("segmentation"))) and not pd.isna(row["seg_area"])
            else np.nan,
            axis=1
        )

        # 4c) Compute ratios only for valid seg_area rows
        df_merged["seg_to_bbox_ratio"] = (
                df_merged["seg_area"] / df_merged["bbox_area"]
        )
        df_merged["seg_to_img_ratio"] = (
                df_merged["seg_area"] / df_merged["img_area"]
        )

    return df_merged


if __name__ == '__main__':
    from config import create_config
    from loader import create_coco_dataframe
    import pandas as pd
    pd.set_option('display.max_columns', None)

    config = create_config()
    df_from_coco = create_coco_dataframe(config)

    df_merged = create_merged_df(df_from_coco)

    print(df_merged.head())
