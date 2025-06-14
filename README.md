# Deep Learning-Based Histomorphological Reclassification and treatment decision optimization for Combined Small Cell and Large Cell Lung Neuroendocrine Carcinomas

## Introduction
![image](https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Figure1B.png#pic_center)

## Datasets
- ​`clinical_data`: Including patient categories, overall survival time/status, and disease-free survival time/status.
- ​`WSIs`: H&E-stained whole-slide images.
- ​`patches`: 224×224 pixel patches generated from WSIs at 5× magnification.

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
  Run ​`tile_WSI.py​` and configure mandatory parameters in script headers for your use case.
  - #### Step Inputs: 
    WSI path: You need to set the path storing your WSIs in the ​`slidepath​` parameter on line 700 of the code.
  - #### Step Outputs:
    The patch file for each WSI: You can configure the output location of the patch with the ​`output​` parameter.
### 2. ​Extract patch feature and build graphs
  ```bash
  python build_graphs.py
  ```
  Go to `build_graphs.py` to use the UNI feature extractor for extracting patch features and constructing graphs. You can replace the feature extractor as needed by modifying the relevant section in `build_graphs.py`.
  - #### Step Inputs: 
    Patch path: You need to set the path storing your patches in the ​`dataset​` parameter.
  - #### Step Outputs:
    Patch features of the graph structure: Configure the storage path of features using the ​`output​` parameter. Each WSI feature folder corresponds to three subfiles: the patch feature file features.pt that stores the WSI, the adjacency matrix of the patch adj_s.pt, and the coordinate file of the patch c_idx.txt.
### 3. ​Model Training
  ```bash
  python main_oriTrain.py
  ```
  Run this script to train the model and automatically save it to the specified path.
  - #### Step Inputs: 
    TXT format file for cross-validation for model training: e.g.: ​`data_demo/train_1.txt​`.
    
    TXT format files for cross-validation for model validation: e.g.: ​`data_demo/val_1.txt​`.
    
    Patch feature path: e.g.: ​`results/features​`.
    
  - #### Step Outputs:
    The model with the best performance on the internal validation set: Stored in the path corresponding to the ​`model_path​` parameter.
    
    Training logs: Stored in the path corresponding to the ​`log_path parameter​`.
    
### 4. ​Model Test
  ```bash
  python main_test_orVisual.py
  ```
  - #### Step Inputs: 
    TXT format files for validation: e.g.: ​`data_demo/external.txt​`.
    
    The path where the patch features of the graph structure used for validation: e.g.: ​`results/features​`.
    
    The storage path for the best model: You can configure the output location of the best model with the ​`resume​` parameter.
    
  - #### Step Outputs:
    The model with the best performance on the internal validation set: Stored in the path corresponding to the ​`model_path​` parameter.
    
    Training logs: Stored in the path corresponding to the ​`log_path​` parameter​.
    
### 5. ​Interpretability
  ```bash
  python visualisation.py
  ```
  Note: Prior to executing python visualisation.py, ensure that the Pillow package is upgraded to version ​`9.5.0​`, and the timm package is upgraded to version ​`1.0.15​`.
  - #### Step Inputs: 
    CSV format files for attention scores for each patch from the model: The CSV should contain N rows and 2 columns (N = number of patches + 1), with the first row containing the headers: "patch_name" and "attention". e.g.: ​`data_demo/WSI1_attention_score.csv`.
    
    The path where the patch features of the graph structure used for validation: e.g.: ​`results/features​`.
    
    Pathology foundation model: e.g.: ​`https://github.com/mahmoodlab/UNI​`.
    
    The storage path for the best model: ​`resume​` parameter.
    
  - #### Step Outputs:
    Attention heatmap results: e.g.: ​`https://github.com/mahmoodlab/UNI​`.
    
  <div align="center">
    <img src="https://github.com/Shengruyuyu/cSCLC-LCNEC/blob/main/images/Figure2D.png" 
         width="70%" 
         style="transform-origin: center; display: block; margin: auto;">
  </div>

## Web-based platform
Click and access our free open platform at [http://lungnecrisk.com](http://lungnecrisk.com) to upload WSI and receive results.   


###### For further reference, you may also refer to our reference implementation available at: https://github.com/vkola-lab/tmi2022
