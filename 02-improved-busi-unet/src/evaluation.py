import numpy as np


def mean_dice(y_true, y_pred, threshold=0.5, eps=1e-6):
    """
    Compute mean foreground Dice coefficient over a dataset.

    Predictions are binarized using the specified threshold and
    Dice is calculated independently for each validation sample.
    """
    y_true = (y_true > 0.5).astype(np.float32)
    y_pred = (y_pred > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))

    denominator = (
        np.sum(y_true, axis=(1, 2, 3))
        + np.sum(y_pred, axis=(1, 2, 3))
    )

    dice = (2.0 * intersection + eps) / (denominator + eps)

    return float(np.mean(dice))


def mean_iou(y_true, y_pred, threshold=0.5, eps=1e-6):
    """
    Compute mean foreground IoU over a dataset.

    Predictions are binarized using the specified threshold and
    IoU is calculated independently for each validation sample.
    """
    y_true = (y_true > 0.5).astype(np.float32)
    y_pred = (y_pred > threshold).astype(np.float32)

    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))

    union = (
        np.sum(y_true, axis=(1, 2, 3))
        + np.sum(y_pred, axis=(1, 2, 3))
        - intersection
    )

    iou = (intersection + eps) / (union + eps)

    return float(np.mean(iou))