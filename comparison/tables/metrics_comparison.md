# Final Baseline vs Improved Comparison

The final comparison uses **Notebook Contract 1.0** and the exact same
**130-case validation cohort** for both models.

## Headline result

The Improved U-Net increases offline mean foreground Dice from **0.6869**
to **0.7004** (**+1.35 percentage points**) and IoU from
**0.5895** to **0.6040** (**+1.45 percentage points**).

| Metric | Baseline U-Net | Improved U-Net | Absolute Δ | Relative improvement |
|---|---:|---:|---:|---:|
| Mean foreground Dice | 0.6869 | **0.7004** | +0.0135 | +1.96% |
| Mean foreground IoU | 0.5895 | **0.6040** | +0.0145 | +2.46% |
| Best monitored soft validation Dice | 0.6593 | **0.6893** | +0.0300 | — |
| Best epoch | 50 | 24 | — | — |
| Epochs completed | 50 | 34 | — | — |

> **Canonical headline metrics are the offline mean per-sample foreground
> Dice and IoU computed from the exact best checkpoint at threshold 0.5.**
> The soft validation Dice is the training monitor, not the headline metric.

## Fair-comparison checks

| Check | Result |
|---|---|
| BUSI experiment cases | 647 in both runs |
| Train / validation split | 517 / 130 in both runs |
| Split seed | 42 |
| Dataset order fingerprint | MATCH |
| Validation membership fingerprint | MATCH |
| Exact validation case IDs | MATCH |
| Prediction threshold | 0.5 in both runs |
| Offline metric definition | Mean per-sample foreground in both runs |
| Ground-truth ROI crop | Disabled in both runs |
| Validation augmentation | Disabled in both runs |
| Best checkpoint reloaded | Yes in both runs |

**Validation SHA-256**

`e5c318a528dee5a335218114b5b1601b5dbae8276686d739126ea5d5c3a8344c`

## Intended experimental differences

| Configuration | Baseline U-Net | Improved U-Net |
|---|---|---|
| Input shape | 128×128×1 | 192×192×1 |
| Dropout | 0.15 | 0.10 |
| Loss | Binary Crossentropy | BCE + Dice |
| Augmentation | None | Training only |
| Train samples before augmentation | 517 | 517 |
| Train samples after augmentation | 517 | 1034 |
| Source commit | `5d10931` | `0cd7317` |

## Interpretation

The Improved U-Net produces a **modest but consistent gain** on the same
held-out validation cohort. The gain is therefore attributable to the changed
model/training recipe under this experiment, rather than to a different
validation split.

Cross-model training-loss values are **not compared directly** because the two
models optimize different loss functions. Model-specific loss curves are kept
under `comparison/baseline/` and `comparison/improved/`; the shared validation
Dice trajectory is used for cross-model training-dynamics comparison.
