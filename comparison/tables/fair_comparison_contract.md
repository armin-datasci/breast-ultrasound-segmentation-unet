# Fair Comparison Contract

- Notebook contract: **1.0**
- Notebook-contract repository commit: **919e74d**
- Dataset: **BUSI**
- Total BUSI cases: **780**
- Experiment cases: **647** (437 benign + 210 malignant; 133 normal excluded)
- Split: **517 train / 130 validation**
- Seed: **42**
- Threshold: **0.5**
- Canonical metrics: **mean per-sample foreground Dice and IoU**
- Dataset-order SHA-256: `bfc46d4ecf1483ba7699fb726f588dee2307e0e46ef77aa690f22857b3a4af1e`
- Validation-membership SHA-256: `e5c318a528dee5a335218114b5b1601b5dbae8276686d739126ea5d5c3a8344c`
- Validation case IDs identical: **yes**
- Ground-truth ROI used for input crop: **no**
- Validation augmentation: **no**
- Evaluation checkpoint: **exact best ModelCheckpoint reloaded**

The only intentional differences are model/training choices: input resolution,
architecture width, dropout, loss, and train-only augmentation.
