# Breast Ultrasound Lesion Segmentation with U-Net

A reproducible, leakage-free comparison of **Baseline** and **Improved U-Net** pipelines for breast-lesion segmentation on the **BUSI (Breast Ultrasound Images) dataset**.

This repository focuses not only on segmentation performance, but also on **fair experimental comparison, deterministic validation, reproducibility, and methodological correctness**.

---

## Key Results

Both models were evaluated on the **exact same 130-image validation cohort** using the same threshold and the same offline metric definitions.

| Model | Mean Foreground Dice | Mean Foreground IoU |
|---|---:|---:|
| Baseline U-Net | 0.6869 | 0.5895 |
| **Improved U-Net** | **0.7004** | **0.6040** |
| **Improvement** | **+1.35 pp** | **+1.45 pp** |

The Improved U-Net achieved a modest but consistent improvement while preserving a leakage-free and reproducible evaluation protocol.

![Final offline validation metrics](comparison/plots/dice_iou_comparison.png)

---

## Tech Stack

**Python · TensorFlow / Keras · NumPy · OpenCV · scikit-learn · Matplotlib · Google Colab · Git · GitHub**

Core areas demonstrated in this project:

- Deep learning for semantic segmentation
- Medical-image preprocessing
- U-Net architecture design
- Custom loss and segmentation metrics
- Data augmentation
- Reproducible experiment design
- Leakage detection and correction
- Best-checkpoint evaluation
- Git-based experiment versioning

---

## Quick Links

| Resource | Link |
|---|---|
| Baseline U-Net Notebook | [Open notebook](01-baseline-busi-unet/notebook/baseline_busi_unet.ipynb) |
| Improved U-Net Notebook | [Open notebook](02-improved-busi-unet/notebook/improved_busi_unet.ipynb) |
| Final Model Comparison | [View comparison](comparison/tables/metrics_comparison.md) |
| Fair Comparison Contract | [View contract](comparison/tables/fair_comparison_contract.md) |
| Comparison Manifest | [View manifest](comparison/comparison_manifest.json) |
| Trained Models Archive | [Download from Google Drive](https://drive.google.com/file/d/11eYIN_sAzJklZiPeKr5EngA2bH6aebyu/view?usp=sharing) |
| BUSI Dataset Mirror | [Download from Google Drive](https://drive.google.com/file/d/1Pl22yxAccHBUfcBCAgMIK1XesDES-Ytl/view?usp=sharing) |

> The Google Drive BUSI file is provided only as a reproducibility mirror.  
> Please cite the original BUSI publication listed in the References section.

---

## Project Highlights

This project demonstrates an end-to-end segmentation workflow with emphasis on experimental reliability:

- Built two modular U-Net segmentation pipelines.
- Standardized both notebooks under a shared **Notebook Contract v1.0**.
- Used deterministic dataset ordering and a shared random seed.
- Evaluated both models on the exact same validation cases.
- Fingerprinted the validation membership using SHA-256.
- Reloaded the exact best saved checkpoint before final evaluation.
- Used mean per-sample foreground Dice and IoU as canonical offline metrics.
- Restricted augmentation to training data only.
- Removed ground-truth-dependent ROI preprocessing to eliminate target leakage.
- Preserved run metadata, training histories, validation manifests, and comparison artifacts.

---

# 1. Project Overview

Breast-lesion segmentation from ultrasound images is a semantic segmentation task in which a model predicts the lesion region at the pixel level.

This repository compares two U-Net-based approaches:

### Baseline U-Net

A smaller reference configuration using:

- 128 × 128 grayscale inputs
- Binary Cross-Entropy loss
- 0.15 dropout
- No data augmentation

### Improved U-Net

A wider configuration using:

- 192 × 192 grayscale inputs
- Wider encoder/decoder feature maps
- BCE + Dice loss
- 0.10 dropout
- Training-only augmentation

The objective is not simply to maximize a metric, but to determine whether these changes improve segmentation performance under a controlled and reproducible evaluation protocol.

---

# 2. Dataset

The project uses the **BUSI — Breast Ultrasound Images Dataset**.

Dataset composition:

| Class | Images |
|---|---:|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| **Total** | **780** |

The segmentation experiment uses:

```text
437 benign
+ 210 malignant
--------------
647 experiment cases
```

The **133 normal cases are excluded** from this binary lesion-segmentation experiment.

The experiment split is:

```text
Training:   517
Validation: 130
Total:      647
```

Both models use exactly the same validation cohort.

---

## Preprocessing

Images and masks are loaded independently.

For both pipelines:

- Images are converted to grayscale.
- Full ultrasound images are resized to the target model resolution.
- Images are normalized to `[0, 1]`.
- Masks are binarized.
- Images use linear interpolation during resizing.
- Masks use nearest-neighbor interpolation to preserve segmentation labels.
- File ordering is deterministic.
- Ground-truth masks are never used to determine an image crop or ROI.

This last condition is particularly important because using the target mask to determine the model input would introduce **target leakage**.

---

# 3. Experimental Design

The final experiments follow **Notebook Contract v1.0**.

Shared conditions include:

| Setting | Value |
|---|---|
| Experiment cases | 647 |
| Training cases before augmentation | 517 |
| Validation cases | 130 |
| Validation fraction | 20% |
| Random seed | 42 |
| Batch size | 8 |
| Maximum epochs | 50 |
| Initial learning rate | 1e-4 |
| Prediction threshold | 0.5 |
| Canonical Dice | Mean per-sample foreground Dice |
| Canonical IoU | Mean per-sample foreground IoU |

The validation membership is fingerprinted with:

```text
SHA-256:
e5c318a528dee5a335218114b5b1601b5dbae8276686d739126ea5d5c3a8344c
```

The identical fingerprint and stored case IDs verify that both models were evaluated on the same 130 validation images.

---

# 4. Model Architectures

Both models follow the U-Net encoder-decoder pattern with skip connections and Batch Normalization.

The main architectural difference is model capacity.

| Configuration | Baseline U-Net | Improved U-Net |
|---|---:|---:|
| Input | 128 × 128 × 1 | 192 × 192 × 1 |
| Encoder filters | 24 → 48 → 96 → 192 | 32 → 64 → 128 → 256 |
| Bottleneck filters | 384 | 512 |
| Dropout | 0.15 | 0.10 |
| Output | 1-channel sigmoid mask | 1-channel sigmoid mask |

---

## Baseline U-Net

The Baseline model establishes the reference experiment.

Configuration:

```text
Input size:       128 × 128 × 1
Loss:             Binary Cross-Entropy
Dropout:          0.15
Batch size:       8
Maximum epochs:   50
Augmentation:     None
Training samples: 517
Validation:       130
```

Final run:

```text
Epochs completed:                   50
Best epoch:                         50
Best monitored soft val Dice:      0.6593
Offline mean foreground Dice:       0.6869
Offline mean foreground IoU:        0.5895
```

---

## Improved U-Net

The Improved model increases input resolution and network width, changes the loss function, and introduces light train-only augmentation.

Configuration:

```text
Input size:       192 × 192 × 1
Loss:             BCE + Dice
Dropout:          0.10
Batch size:       8
Maximum epochs:   50
Training before augmentation: 517
Training after augmentation:  1034
Validation:                  130
```

Training-only augmentation includes:

- Horizontal flip
- Vertical flip
- Rotation up to ±15°
- Small intensity changes

No augmentation is applied to validation data.

Final run:

```text
Epochs completed:                   34
Best epoch:                         24
Best monitored soft val Dice:      0.6893
Offline mean foreground Dice:       0.7004
Offline mean foreground IoU:        0.6040
```

---

# 5. Final Comparison

The canonical comparison is based on predictions generated by the **exact best saved checkpoint**, reloaded from disk after training.

| Metric | Baseline | Improved | Change |
|---|---:|---:|---:|
| Mean Foreground Dice | 0.6869 | **0.7004** | **+0.0135** |
| Mean Foreground IoU | 0.5895 | **0.6040** | **+0.0145** |
| Best soft validation Dice | 0.6593 | **0.6893** | +0.0300 |
| Best epoch | 50 | 24 | — |
| Epochs completed | 50 | 34 | — |

The headline metrics are the **offline mean foreground Dice and IoU**, not the soft Dice used during training monitoring.

---

## Training Dynamics

![Validation Dice comparison](comparison/plots/validation_dice_comparison.png)

The Improved model reaches its best validation Dice earlier and training stops after validation performance ceases to improve.

Training-loss curves are stored separately for each model:

- [`comparison/baseline/training_loss.png`](comparison/baseline/training_loss.png)
- [`comparison/improved/training_loss.png`](comparison/improved/training_loss.png)

The two loss values are **not compared directly** because the models optimize different objective functions:

```text
Baseline → Binary Cross-Entropy

Improved → Binary Cross-Entropy + Dice Loss
```

A direct numerical comparison between these training losses would therefore be misleading.

---

# 6. Methodological Integrity

A central part of this project was improving the reliability of the experimental pipeline.

During refactoring, an earlier preprocessing approach that derived a region of interest from the ground-truth mask was identified as a form of **target leakage**.

That approach was removed.

The final pipeline:

```text
Ultrasound image
      │
      ▼
Full-image resizing
      │
      ▼
Normalization
      │
      ▼
U-Net
      │
      ▼
Predicted segmentation mask
```

The ground-truth mask is used only as the supervised training/evaluation target.

It is never used to determine the model input.

For this reason, earlier results produced by the leakage-prone pipeline are not reported as final project results.

---

# 7. Reproducibility

The Baseline and Improved notebooks follow the same high-level execution structure:

```text
1. Runtime and reproducibility setup
2. Dataset download and extraction
3. Dataset validation
4. Source and methodology checks
5. Experiment configuration
6. Preprocessing
7. Train / validation split
8. Training-data preparation
9. Model construction
10. Training
11. Best-checkpoint evaluation
12. Results and artifacts
13. Final run summary
```

Each final run produces:

```text
best_model.keras
run_metadata.json
training_history.json
validation_split.json
figures/
├── training_loss.png
└── validation_metrics.png
```

Large trained-model files are intentionally excluded from Git and are available through the **Trained Models Archive** linked near the top of this README.

---

## Reproducing the Experiments

The recommended environment is **Google Colab with GPU acceleration**.

### Baseline

Open:

```text
01-baseline-busi-unet/notebook/baseline_busi_unet.ipynb
```

Start a fresh runtime and execute:

```text
Runtime → Run all
```

### Improved

Use a separate fresh runtime and open:

```text
02-improved-busi-unet/notebook/improved_busi_unet.ipynb
```

Again execute:

```text
Runtime → Run all
```

Both notebooks download the same BUSI dataset mirror and reconstruct the deterministic experiment split.

After both runs, the validation fingerprints should match.

---

## Regenerating the Comparison

The comparison directory acts as the source of truth for the final model comparison.

With the required Python dependencies installed:

```bash
python comparison/generate_comparison.py
```

The script validates the stored experiment metadata and regenerates the comparison artifacts.

Expected headline result:

```text
Dice: 0.6869 → 0.7004
IoU:  0.5895 → 0.6040
Validation fingerprint match: True
```

---

# 8. Repository Structure

```text
breast-ultrasound-segmentation-unet/
│
├── 01-baseline-busi-unet/
│   ├── notebook/
│   │   └── baseline_busi_unet.ipynb
│   │
│   ├── src/
│   │   ├── callbacks.py
│   │   ├── dataset_loader.py
│   │   ├── evaluation.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   ├── train_pipeline.py
│   │   └── utils.py
│   │
│   └── figures/
│       ├── training_loss.png
│       └── validation_metrics.png
│
├── 02-improved-busi-unet/
│   ├── notebook/
│   │   └── improved_busi_unet.ipynb
│   │
│   ├── src/
│   │   ├── augmentation.py
│   │   ├── callbacks.py
│   │   ├── dataset_loader.py
│   │   ├── evaluation.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── train_pipeline.py
│   │   └── utils.py
│   │
│   └── figures/
│       ├── training_loss.png
│       └── validation_metrics.png
│
├── comparison/
│   ├── baseline/
│   │   ├── run_metadata.json
│   │   ├── training_history.json
│   │   ├── validation_split.json
│   │   ├── training_loss.png
│   │   └── validation_metrics.png
│   │
│   ├── improved/
│   │   ├── run_metadata.json
│   │   ├── training_history.json
│   │   ├── validation_split.json
│   │   ├── training_loss.png
│   │   └── validation_metrics.png
│   │
│   ├── plots/
│   │   ├── dice_iou_comparison.png
│   │   └── validation_dice_comparison.png
│   │
│   ├── tables/
│   │   ├── fair_comparison_contract.md
│   │   ├── metrics_comparison.csv
│   │   └── metrics_comparison.md
│   │
│   ├── comparison_manifest.json
│   └── generate_comparison.py
│
├── .gitignore
├── LICENSE
└── README.md
```

---

# 9. What This Project Demonstrates

From a machine-learning engineering perspective, the project demonstrates:

### Modeling

- Encoder-decoder CNN architectures
- U-Net skip connections
- Binary semantic segmentation
- Custom segmentation losses
- Training regularization
- Data augmentation

### Data & Evaluation

- Image/mask preprocessing
- Correct interpolation for segmentation labels
- Deterministic sample ordering
- Controlled train/validation splitting
- Per-sample Dice and IoU evaluation
- Best-checkpoint evaluation

### Reproducibility

- Fixed random seed
- Shared experiment contract
- Validation-set fingerprinting
- Machine-readable run metadata
- Reproducible comparison generation
- Version-controlled source snapshots

### Engineering Practice

- Modular Python source code
- Experiment separation
- Git-based development
- Artifact management
- Detection and correction of methodological leakage
- Documentation designed for reproducible review

---

# 10. Limitations

This project is intentionally scoped as a controlled academic and portfolio experiment.

Important limitations include:

- Evaluation is performed on a single public dataset.
- Only 647 benign and malignant cases are used for lesion segmentation.
- The final evaluation uses a held-out validation split rather than an independent external test dataset.
- Cross-validation is not performed.
- Normal BUSI cases are outside the segmentation experiment.
- The performance improvement of the Improved model is modest.
- The models have not been clinically validated.
- Results should not be interpreted as evidence of clinical diagnostic performance.

This repository is intended for **educational and research purposes**, not for clinical use.

---

# 11. Academic Context

This repository is a refactored and reproducible portfolio implementation developed from a Bachelor's-level project on medical-image segmentation.

The final version places particular emphasis on:

- reproducibility,
- fair model comparison,
- leakage-free methodology,
- modular code,
- transparent reporting of results.

---

# 12. References

### U-Net

Ronneberger, O., Fischer, P., & Brox, T. (2015).

**U-Net: Convolutional Networks for Biomedical Image Segmentation.**

arXiv:1505.04597

### BUSI Dataset

Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A.

**Dataset of Breast Ultrasound Images.**

*Data in Brief*, Volume 28, 104863.

DOI: `10.1016/j.dib.2019.104863`

---

## Final Takeaway

Under an identical validation cohort and standardized evaluation protocol:

```text
Baseline U-Net
Dice = 0.6869
IoU  = 0.5895

        ↓

Improved U-Net
Dice = 0.7004
IoU  = 0.6040
```

The main contribution of this repository is therefore not only the metric improvement, but a **reproducible, leakage-free, and auditable workflow for comparing segmentation models fairly**.
