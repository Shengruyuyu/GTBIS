# Deep Learning-Based Histomorphological Reclassification and Risk Stratification of Combined Small-Cell and Large-Cell Neuroendocrine Carcinomas

## Introduction
![image](https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Images_Figure1B.png)
*study design*

## Datasets
- ​`clinical_data`: Contains three-cohort clinical metadata including patient categories, overall survival time/status, and disease-free survival time/status
- ​`WSIs`: Whole-slide images from three independent cohorts
- ​`patches`: 224×224 pixel patches generated from WSIs at 5× magnification

## Model Overview
![image](https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Images_Figure1C.png)
*Overview of model architecture*

## Installation
```bash
git clone https://......
cd your-repo-name
```

## Usage
- ### ​Tiling patch
  ```bash
  python src/tile_WSI.py
  ```
  Run src/tile_WSI.py and configure mandatory parameters in script headers for your use case.
- ### ​Extract patch feature and build graphs
  ```bash
  python feature_extractor_U/build_graphs.py
  ```
  Go to `./feature_extractor_U` to use the UNI feature extractor for extracting patch features and constructing graphs. You can replace the feature extractor as needed by modifying the relevant section in `./feature_extractor_U/build_graphs.py`.
- ### ​Model Training
  ```bash
  python main_oriTrain.py
  ```
  Run this script to train the model and automatically save it to the specified path.
- ### ​Interpretability
  ```bash
  python main_oriTrain.py
  ```
  Before running `main_oriTrain.py`, obtain the model's attention scores for each patch and save them in a `.csv` file. When `main_oriTrain.py` is executed, it will read the `.csv` file and integrate the deep features from the UNI feature     extractor to generate the final "pixel-level" heatmap.
  
  
  *Morphological characteristics visualization*
