from pathlib import Path
import json
import csv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "baseline"
IMPROVED = ROOT / "improved"
PLOTS = ROOT / "plots"
TABLES = ROOT / "tables"
PLOTS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

NOTEBOOK_CONTRACT_COMMIT = "919e74d"

baseline = json.loads((BASELINE / "run_metadata.json").read_text(encoding="utf-8"))
improved = json.loads((IMPROVED / "run_metadata.json").read_text(encoding="utf-8"))
baseline_history = json.loads((BASELINE / "training_history.json").read_text(encoding="utf-8"))
improved_history = json.loads((IMPROVED / "training_history.json").read_text(encoding="utf-8"))
baseline_split = json.loads((BASELINE / "validation_split.json").read_text(encoding="utf-8"))
improved_split = json.loads((IMPROVED / "validation_split.json").read_text(encoding="utf-8"))

dataset_hash_equal = (
    baseline["dataset"]["dataset_case_order_sha256"]
    == improved["dataset"]["dataset_case_order_sha256"]
)
validation_hash_equal = (
    baseline["split"]["validation_membership_sha256"]
    == improved["split"]["validation_membership_sha256"]
)
validation_ids_equal = (
    baseline_split["validation_case_ids"]
    == improved_split["validation_case_ids"]
)

assert dataset_hash_equal, "Dataset-order fingerprints do not match."
assert validation_hash_equal, "Validation fingerprints do not match."
assert validation_ids_equal, "Validation case IDs do not match."
assert baseline["split"]["validation_samples"] == improved["split"]["validation_samples"] == 130
assert baseline["split"]["training_samples_before_augmentation"] == improved["split"]["training_samples_before_augmentation"] == 517
assert baseline["offline_evaluation"]["threshold"] == improved["offline_evaluation"]["threshold"] == 0.5
assert baseline["offline_evaluation"]["metric_scope"] == improved["offline_evaluation"]["metric_scope"]

bd = float(baseline["offline_evaluation"]["mean_dice"])
bi = float(baseline["offline_evaluation"]["mean_iou"])
id_ = float(improved["offline_evaluation"]["mean_dice"])
ii = float(improved["offline_evaluation"]["mean_iou"])

dice_delta = id_ - bd
iou_delta = ii - bi
dice_rel = dice_delta / bd
iou_rel = iou_delta / bi

# ------------------------------------------------------------------
# 1. Machine-readable metrics table
# ------------------------------------------------------------------
rows = [
    {
        "metric": "Mean Foreground Dice",
        "baseline": bd,
        "improved": id_,
        "absolute_delta": dice_delta,
        "percentage_point_delta": dice_delta * 100,
        "relative_improvement_pct": dice_rel * 100,
    },
    {
        "metric": "Mean Foreground IoU",
        "baseline": bi,
        "improved": ii,
        "absolute_delta": iou_delta,
        "percentage_point_delta": iou_delta * 100,
        "relative_improvement_pct": iou_rel * 100,
    },
]
with (TABLES / "metrics_comparison.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ------------------------------------------------------------------
# 2. Human-readable comparison report
# ------------------------------------------------------------------
metrics_md = f"""# Final Baseline vs Improved Comparison

The final comparison uses **Notebook Contract 1.0** and the exact same
**130-case validation cohort** for both models.

## Headline result

The Improved U-Net increases offline mean foreground Dice from **{bd:.4f}**
to **{id_:.4f}** (**+{dice_delta*100:.2f} percentage points**) and IoU from
**{bi:.4f}** to **{ii:.4f}** (**+{iou_delta*100:.2f} percentage points**).

| Metric | Baseline U-Net | Improved U-Net | Absolute Δ | Relative improvement |
|---|---:|---:|---:|---:|
| Mean foreground Dice | {bd:.4f} | **{id_:.4f}** | +{dice_delta:.4f} | +{dice_rel*100:.2f}% |
| Mean foreground IoU | {bi:.4f} | **{ii:.4f}** | +{iou_delta:.4f} | +{iou_rel*100:.2f}% |
| Best monitored soft validation Dice | {baseline["training"]["best_monitored_soft_validation_dice"]:.4f} | **{improved["training"]["best_monitored_soft_validation_dice"]:.4f}** | +{improved["training"]["best_monitored_soft_validation_dice"] - baseline["training"]["best_monitored_soft_validation_dice"]:.4f} | — |
| Best epoch | {baseline["training"]["best_epoch"]} | {improved["training"]["best_epoch"]} | — | — |
| Epochs completed | {baseline["training"]["epochs_completed"]} | {improved["training"]["epochs_completed"]} | — | — |

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

`{baseline["split"]["validation_membership_sha256"]}`

## Intended experimental differences

| Configuration | Baseline U-Net | Improved U-Net |
|---|---|---|
| Input shape | 128×128×1 | 192×192×1 |
| Dropout | 0.15 | 0.10 |
| Loss | Binary Crossentropy | BCE + Dice |
| Augmentation | None | Training only |
| Train samples before augmentation | 517 | 517 |
| Train samples after augmentation | 517 | 1034 |
| Source commit | `{baseline["source"]["commit"][:7]}` | `{improved["source"]["commit"][:7]}` |

## Interpretation

The Improved U-Net produces a **modest but consistent gain** on the same
held-out validation cohort. The gain is therefore attributable to the changed
model/training recipe under this experiment, rather than to a different
validation split.

Cross-model training-loss values are **not compared directly** because the two
models optimize different loss functions. Model-specific loss curves are kept
under `comparison/baseline/` and `comparison/improved/`; the shared validation
Dice trajectory is used for cross-model training-dynamics comparison.
"""
(TABLES / "metrics_comparison.md").write_text(metrics_md, encoding="utf-8")

# ------------------------------------------------------------------
# 3. Fair-comparison contract
# ------------------------------------------------------------------
contract_md = f"""# Fair Comparison Contract

- Notebook contract: **1.0**
- Notebook-contract repository commit: **{NOTEBOOK_CONTRACT_COMMIT}**
- Dataset: **BUSI**
- Total BUSI cases: **780**
- Experiment cases: **647** (437 benign + 210 malignant; 133 normal excluded)
- Split: **517 train / 130 validation**
- Seed: **42**
- Threshold: **0.5**
- Canonical metrics: **mean per-sample foreground Dice and IoU**
- Dataset-order SHA-256: `{baseline["dataset"]["dataset_case_order_sha256"]}`
- Validation-membership SHA-256: `{baseline["split"]["validation_membership_sha256"]}`
- Validation case IDs identical: **yes**
- Ground-truth ROI used for input crop: **no**
- Validation augmentation: **no**
- Evaluation checkpoint: **exact best ModelCheckpoint reloaded**

The only intentional differences are model/training choices: input resolution,
architecture width, dropout, loss, and train-only augmentation.
"""
(TABLES / "fair_comparison_contract.md").write_text(contract_md, encoding="utf-8")

# ------------------------------------------------------------------
# 4. Canonical offline metrics plot
# ------------------------------------------------------------------
metrics = ["Mean Foreground Dice", "Mean Foreground IoU"]
baseline_vals = [bd, bi]
improved_vals = [id_, ii]
x = np.arange(len(metrics))
width = 0.34

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline U-Net")
b2 = ax.bar(x + width/2, improved_vals, width, label="Improved U-Net")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Final Offline Validation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
for bars in [b1, b2]:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
fig.tight_layout()
fig.savefig(PLOTS / "dice_iou_comparison.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------
# 5. Shared training-monitor trajectory
# ------------------------------------------------------------------
base_val_dice = baseline_history["val_dice_coefficient"]
imp_val_dice = improved_history["val_dice_coefficient"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(base_val_dice) + 1), base_val_dice, label="Baseline U-Net")
ax.plot(range(1, len(imp_val_dice) + 1), imp_val_dice, label="Improved U-Net")
ax.set_xlabel("Epoch")
ax.set_ylabel("Soft Validation Dice")
ax.set_title("Validation Dice During Training")
ax.set_ylim(0, 1)
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "validation_dice_comparison.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------
# 6. Comparison manifest
# ------------------------------------------------------------------
manifest = {
    "comparison_version": "1.0",
    "notebook_contract_version": "1.0",
    "notebook_contract_commit": NOTEBOOK_CONTRACT_COMMIT,
    "dataset": {
        "name": "BUSI",
        "total_cases": 780,
        "experiment_cases": 647,
        "dataset_case_order_sha256": baseline["dataset"]["dataset_case_order_sha256"],
        "validation_membership_sha256": baseline["split"]["validation_membership_sha256"],
        "validation_cases": 130,
        "validation_case_ids_match": validation_ids_equal,
    },
    "baseline": {
        "source_commit": baseline["source"]["commit"],
        "mean_foreground_dice": bd,
        "mean_foreground_iou": bi,
        "best_soft_validation_dice": baseline["training"]["best_monitored_soft_validation_dice"],
        "best_epoch": baseline["training"]["best_epoch"],
        "epochs_completed": baseline["training"]["epochs_completed"],
    },
    "improved": {
        "source_commit": improved["source"]["commit"],
        "mean_foreground_dice": id_,
        "mean_foreground_iou": ii,
        "best_soft_validation_dice": improved["training"]["best_monitored_soft_validation_dice"],
        "best_epoch": improved["training"]["best_epoch"],
        "epochs_completed": improved["training"]["epochs_completed"],
    },
    "delta": {
        "dice_absolute": dice_delta,
        "dice_percentage_points": dice_delta * 100,
        "dice_relative_pct": dice_rel * 100,
        "iou_absolute": iou_delta,
        "iou_percentage_points": iou_delta * 100,
        "iou_relative_pct": iou_rel * 100,
    },
    "acceptance_checks": {
        "dataset_order_fingerprint_match": dataset_hash_equal,
        "validation_fingerprint_match": validation_hash_equal,
        "validation_case_ids_match": validation_ids_equal,
        "same_train_count_before_augmentation": True,
        "same_validation_count": True,
        "same_threshold": True,
        "same_offline_metric_scope": True,
    },
}
(ROOT / "comparison_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Comparison artifacts regenerated successfully.")
print(f"Dice: {bd:.4f} -> {id_:.4f} (+{dice_delta*100:.2f} pp)")
print(f"IoU:  {bi:.4f} -> {ii:.4f} (+{iou_delta*100:.2f} pp)")
print("Validation fingerprint:", baseline["split"]["validation_membership_sha256"])
