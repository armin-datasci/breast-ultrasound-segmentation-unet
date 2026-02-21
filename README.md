# Bachelor Project: BUSI U-Net Segmentation

---

## Abstract

This Bachelor's project focuses on medical image segmentation using convolutional neural networks applied to the BUSI (Breast Ultrasound Images) dataset. The main objective is to compare a baseline and an improved U-Net model in terms of segmentation performance, reproducibility, and methodological soundness. Key contributions include ROI-based preprocessing, data augmentation, and the combination of Binary Cross-Entropy and Dice loss to improve segmentation accuracy. The repository provides fully structured code, comparative evaluation, and results suitable for academic assessment.

---

## Introduction

Medical image segmentation is a critical task for assisting in the diagnosis of breast lesions. Ultrasound imaging is widely used due to its safety and accessibility, but manual segmentation is time-consuming and subjective. This project implements and compares two U-Net-based architectures for automated segmentation of lesions from ultrasound images. 

The project emphasizes:

- Clear methodological distinction between baseline and improved models  
- Reproducibility of experiments with consistent dataset splits  
- Quantitative evaluation using Dice coefficient and Mean Intersection over Union  
- Code organization  

---

## 1. Project Objective

The primary objective is to segment breast lesions accurately while systematically evaluating how architectural changes, preprocessing, and hyperparameter tuning influence model performance.

---

## 2. Dataset

- **Dataset Name:** BUSI (Breast Ultrasound Images)  
- **Modality:** Grayscale ultrasound images  
- **Annotations:** Binary segmentation masks  
- **Task:** Lesion segmentation (benign vs. malignant)  

All experiments use the same dataset split for fair comparison between models.

---

## 3. Repository Structure

bachelor-project
  - 01-baseline-busi-unet/
    - notebook/
    - src/
    - figures/
  - 02-improved-busi-unet/
    - notebook/
    - src/
    - figures/
  - comparison/
    - plots/
    - tables/
  - README.md
  - LICENSE
  - .gitignore

---

## 4. Methodology

### 4.1 Baseline Model

- Architecture: Original U-Net encoder–decoder  
- Input resolution: 128 × 128  
- Loss function: Binary Cross-Entropy (BCE)  
- Batch size: 8  
- Epochs: 50  
- Dropout: 0.15  
- Preprocessing: None  
- Data augmentation: None  

The baseline serves as a reference implementation to establish a performance lower bound.

---

### 4.2 Improved Model

- Architecture: Enhanced U-Net with wider feature maps and batch normalization  
- Input resolution: 192 × 192  
- Loss function: BCE + Dice loss  
- Batch size: 8  
- Epochs: 50  
- Dropout: 0.10  
- Preprocessing: ROI-based cropping to focus on lesion regions  
- Data augmentation: Light spatial and intensity transformations  

The improved model aims to enhance segmentation accuracy while keeping training conditions comparable to the baseline.

---

## 5. Model Configuration & Hyperparameters

The following hyperparameters are critical to model performance:

- **Input Size:** Larger input (192×192) in the improved model allows better lesion feature representation.  
- **Dropout Rate:** Regularizes the network; lower dropout in the improved model preserves more features while preventing overfitting.  
- **Batch Size:** Controls gradient estimation stability; both models use 8 to balance GPU memory and convergence.  
- **Learning Rate:** 1e-4 in both models ensures stable optimization without overshooting minima.  
- **Epochs:** Set to 50 to allow sufficient training while avoiding overfitting.  

This configuration allows controlled comparison while exploring the impact of architectural and preprocessing improvements.

---

## 6. Evaluation Metrics

Model performance is evaluated using:

- **Dice Coefficient**  
- **Mean Intersection over Union (Mean IoU)**  

All evaluations are conducted on a held-out validation set using identical splits for both models.

---

## 7. Comparative Results

All comparative results are stored in the `comparison/` directory, including plots and tables.

### Example Summary Table

| Model     | Input Size | Dropout | Batch Size | Epochs | Loss Function | Validation Dice | Validation Mean IoU |
|-----------|------------|---------|------------|--------|---------------|-----------------|---------------------|
| Baseline  | 128×128    | 0.15    | 8          | 50     | BCE           | 0.62            | 0.75                |
| Improved  | 192×192    | 0.10    | 8          | 50     | BCE + Dice    | 0.92            | 0.86                |


---

## 8. Reproducibility

### 8.1 Installation


### 8.2 Execution Order

1. Run the baseline notebook:
   `01-baseline-busi-unet/notebook/`  
2. Run the improved notebook:
   `02-improved-busi-unet/notebook/`  
3. Review comparative outputs in:
   `comparison/plots/` and `comparison/tables/`

---

## 9. Academic Context

This project demonstrates:

- Deep learning applied to medical imaging  
- Systematic experimental methodology  
- Clear baseline vs. improved model comparison  
- Code organization and documentation

---

## 10. References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015).  
   *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
   arXiv:1505.04597

2. BUSI Dataset: Breast Ultrasound Images  
   https://www.kaggle.com/datasets/navoneel/breast-ultrasound-images-dataset
