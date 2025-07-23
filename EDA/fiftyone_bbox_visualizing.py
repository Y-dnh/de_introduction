import fiftyone as fo


def launcg_fiftyone_bbox_visualizing(config):

    data_path= config.dataset_dir / 'test'
    labels_path= config.dataset_dir / "annotations/instances_test.json"

    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )

    session = fo.launch_app(dataset, port=5151)
    session.wait()

if __name__ == '__main__':
    from config import create_config

    config = create_config()
    launcg_fiftyone_bbox_visualizing(config)

# FilterLabels, detection, ..., True, False
# {
#   "$and": [
#     {
#       "$gte": [
#         "$$this.bbox_area",
#         0.002
#       ]
#     },
#     {
#       "$lte": [
#         "$$this.bbox_area",
#         0.0025
#       ]
#     }
#   ]
# }