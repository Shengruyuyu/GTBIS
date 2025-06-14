# Deep Learning-Based Histomorphological Reclassification and treatment decision optimization for Combined Small Cell and Large Cell Lung Neuroendocrine Carcinomas

## Introduction
![image](https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Figure1B.png#pic_center)

## Datasets
- ​`clinical_data`: Contains three-cohort clinical metadata including patient categories, overall survival time/status, and disease-free survival time/status
- ​`WSIs`: Whole-slide images from three independent cohorts
- ​`patches`: 224×224 pixel patches generated from WSIs at 5× magnification

## Model Overview
![image](https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Figure1C.png#pic_center)

## Installation
### 1. Download and open our repository
```bash
git clone https://......
cd your-repo-name
```
### 2. Install dependencies
```bash
conda create -n youre_env_name python=3.8 -y
conda activate your_env_name
pip install requirement.txt
```
  
## Usage
### 1. ​Tiling patch
  ```bash
  python tile_WSI.py
  ```
  Run src/tile_WSI.py and configure mandatory parameters in script headers for your use case.
  - #### Step Inputs: 
    WSI path: You need to set the path to store your WSIs in the "slidepath" parameter on line 700 of the code.
  - #### Step Outputs:
    The patch file for each WSI: You can configure the output location of the patch with the --output parameter.
### 2. ​Extract patch feature and build graphs
  ```bash
  python build_graphs.py
  ```
  Go to `build_graphs.py` to use the UNI feature extractor for extracting patch features and constructing graphs. You can replace the feature extractor as needed by modifying the relevant section in `build_graphs.py`.
### 3. ​Model Training
  ```bash
  python main_oriTrain.py
  ```
  Run this script to train the model and automatically save it to the specified path.
### 4. ​Model Test
  ```bash
  python main_test_orVisual.py
  ```
### 5. ​Interpretability
  ```bash
  python visualisation.py
  ```
  Before running visualisation.py, obtain the attention scores for each patch from the model and save them in a .csv file. The CSV should contain N rows and 2 columns (N = number of patches + 1), with the first row containing the headers: "patch_name" and "attention". When visualisation.py is executed, it will read the CSV file and combine the attention scores with deep features extracted by the UNI feature extractor to generate a final pixel-level heatmap.
Note: Prior to executing python visualisation.py, ensure that the Pillow package is upgraded to version 9.5.0, and the timm package is upgraded to version 1.0.15.
  <div align="center">
    <img src="https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Figure2D.png" 
         width="70%" 
         style="transform-origin: center; display: block; margin: auto;">
  </div>

## Web-based platform
Click and access our free open platform at [http://lungnecrisk.com](http://lungnecrisk.com) to upload WSI and receive results.
