# Custom YOLOv8 Object Detection for Person, Pet, and Car

![Python 3.9+](https://img.shields.io/badge/python-3.9-blue.svg)
![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet)

This repository contains a complete pipeline for training, evaluating, and analyzing a custom YOLOv8 object detection model. The project focuses on detecting three specific classes: **Person, Pet (Cat, Dog), and Car (Car, Bus, Truck)**.

A key challenge addressed in this project is the creation of a high-quality, balanced dataset. The standard COCO dataset, while extensive, lacked a sufficient number of 'pet' examples for our needs. To overcome this, we developed a hybrid dataset strategy:

1.  **Training Data:** A custom dataset was curated by selectively merging and sampling from the **COCO 2017** dataset to ensure a balanced representation of our target classes.
2.  **Testing Data:** Since the COCO test set lacks public annotations, we use the **Pascal VOC 2012** dataset, converted to COCO format, as a robust and independent test set for reliable model evaluation.

The entire pipeline, from data preparation to model analysis, is designed to be modular, configurable, and reproducible.

## Key Features

* **Custom Dataset Curation**: Scripts to process, filter, and merge large datasets like COCO and Pascal VOC into a format suitable for custom training.
* **Advanced Data Filtering**: The training pipeline intelligently filters annotations based on bounding box size (min/max area thresholds) and the `iscrowd` flag to improve model performance by removing noisy or uninformative labels.
* **Flexible YOLOv8 Training**: A configurable training notebook (`yolo-training.ipynb`) that allows for easy adjustment of hyperparameters like epochs, batch size, image size, and data filtering rules.
* **Comprehensive Model Evaluation**: A robust testing script (`yolo-testing.ipynb`) that evaluates multiple trained models on the custom Pascal VOC test set, generating detailed performance metrics (mAP, FPS), comparison plots, and sample detection images.
* **In-Depth Dataset Analysis**: A powerful, standalone `dataset_analyzer.py` tool and an `EDA` package to perform Exploratory Data Analysis on any COCO-formatted dataset, generating statistical reports and visualizations.
* **Automated Reporting**: The training and testing pipelines automatically generate detailed `README.md` files and summary reports, ensuring that each experiment is well-documented.

---

### Workflow

The project follows a clear, step-by-step machine learning workflow:

1. **Dataset Preparation**: 
* Run `notebooks/coco-split.ipynb` to generate the training and validation sets from the COCO dataset. This script merges `train2017` and `val2017`, remaps categories (`cat`, `dog` -> `pet`), and samples a balanced dataset. 
* Run `notebooks/pascalvoc-split.ipynb` to convert the Pascal VOC dataset into COCO format. This will serve as the `test` set.

2. **Model Training**: 
* Use `notebooks/yolo-training.ipynb` to train the YOLOv8 model. 
* Configure the training parameters within the notebook, especially the bounding box area thresholds (`BBOX_AREA_TRESHOLD_MIN`, `BBOX_AREA_TRESHOLD_MAX`) and the `SKIP_CROWD_IMAGES` flag to control the data quality. 
* The notebook will output trained model weights (`.pt` files) and a detailed training report.

3. **Model Evaluation**: 
* Run `notebooks/yolo-testing.ipynb` to assess the performance of the trained models. 
* The notebook uses the Pascal VOC test set and generates a comprehensive report comparing all models on metrics like mAP@0.5, mAP@0.5:0.95, and FPS. It also saves visualizations and sample predictions.

4. **Data set analysis (optional)**:
* The `eda/` package can be used to perform detailed analysis of any of the generated data sets. Run `eda/pipeline.py` to generate visualizations and statistics.
* Use the CLI tool `tools/dataset_analyzer.py` for quick analysis of any dataset in COCO format. This is most useful for exploring different variations of box sizes. To generate a readme.md for each dataset, run cmd -> python dataset_analyzer.py <your_dataset_root>

---

###  How to Use

#### Prerequisites

* **Conda (Recommended)**: Create and activate the environment using the `environment.yml` file. This is the recommended approach as it handles all dependencies, including PyTorch and CUDA.
    ```bash
    conda env create -f environment.yml
    conda activate yolo-object-detection-env
    ```

* **Pip**: If you are not using Conda, you can install the required packages using `requirements.txt`. You will need to ensure you have a compatible PyTorch version with CUDA support installed separately.
    ```bash
    pip install -r requirements.txt
    ```


## Key Methodologies

### Hybrid Dataset Strategy

To ensure the model is both well-trained and rigorously tested, we employ a two-dataset approach:
* **COCO for Training**: Leverages the diversity and scale of COCO for robust feature learning. We specifically sample to create a balanced class distribution for `person`, `pet`, and `car`.
* **Pascal VOC for Testing**: Provides a completely independent test set with reliable annotations, preventing any data leakage from the training set and giving a true measure of the model's generalization capabilities.

### Intelligent Data Filtering

Model performance is highly dependent on data quality. Our training pipeline includes two critical filtering steps:
1.  **Bounding Box Area Filtering**: We remove annotations that are either too small (often just noise) or too large (where the object covers most of the image, offering little contextual information). This is controlled by `BBOX_AREA_TRESHOLD_MIN` and `BBOX_AREA_TRESHOLD_MAX`.
2.  **Crowd Annotation Filtering**: Images with `iscrowd=1` annotations are optionally skipped. These annotations mark groups of objects, which can be ambiguous and may not be ideal for training a precise detector.


## Acknowledgments

This work would not be possible without the following outstanding open-source projects and datasets:

* **[Ultralytics](https://ultralytics.com/)**: For the powerful and easy-to-use YOLOv8 framework.
* **[Microsoft COCO: Common Objects in Context](https://cocodataset.org/)**: For providing a diverse and large-scale training dataset.
* **[The PASCAL Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)**: For serving as a robust and independent test set.
