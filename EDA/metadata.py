def create_merged_df(df):

    df["img_area"] = df["width"] * df["height"]
    df = df.rename(columns={"area": "anns_area"})

    return df



if __name__ == '__main__':
    from config import create_config
    from loader import create_coco_dataframe

    config = create_config()
    df = create_coco_dataframe(config)

    metadata = create_merged_df(df)
