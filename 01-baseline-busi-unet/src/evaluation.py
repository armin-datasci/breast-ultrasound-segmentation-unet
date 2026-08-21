import numpy as np


def mean_dice(
    y_true,
    y_pred,
    threshold=0.5,
    eps=1e-6,
):
    """
    Compute mean per-sample foreground Dice coefficient.

    Ground-truth masks are binarized at 0.5 and predictions are
    thresholded before evaluation.
    """

    y_true = (
        y_true > 0.5
    ).astype(np.float32)

    y_pred = (
        y_pred > threshold
    ).astype(np.float32)

    intersection = np.sum(
        y_true * y_pred,
        axis=(1, 2, 3),
    )

    denominator = (
        np.sum(
            y_true,
            axis=(1, 2, 3),
        )
        +
        np.sum(
            y_pred,
            axis=(1, 2, 3),
        )
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return float(
        np.mean(dice)
    )


def mean_iou(
    y_true,
    y_pred,
    threshold=0.5,
    eps=1e-6,
):
    """
    Compute mean per-sample foreground IoU.

    Ground-truth masks are binarized at 0.5 and predictions are
    thresholded before evaluation.
    """

    y_true = (
        y_true > 0.5
    ).astype(np.float32)

    y_pred = (
        y_pred > threshold
    ).astype(np.float32)

    intersection = np.sum(
        y_true * y_pred,
        axis=(1, 2, 3),
    )

    union = (
        np.sum(
            y_true,
            axis=(1, 2, 3),
        )
        +
        np.sum(
            y_pred,
            axis=(1, 2, 3),
        )
        -
        intersection
    )

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return float(
        np.mean(iou)
    )