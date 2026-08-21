import os

import cv2
import numpy as np


def load_busi_dataset(
    path,
    img_size=(128, 128),
):
    """
    Load the BUSI benign and malignant cases for the baseline experiment.

    The loader uses deterministic file ordering and preserves binary
    ground-truth masks during resizing.

    Normal BUSI cases are intentionally excluded from this experiment.
    """

    images = []
    masks = []

    img_height, img_width = img_size

    folders = [
        "benign",
        "malignant",
    ]

    for folder in folders:

        folder_path = os.path.join(
            path,
            folder,
        )

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Missing BUSI class directory: {folder_path}"
            )

        image_files = sorted(
            [
                file
                for file in os.listdir(folder_path)
                if file.endswith(".png")
                and "_mask" not in file
            ]
        )

        for file in image_files:

            img_path = os.path.join(
                folder_path,
                file,
            )

            mask_path = os.path.join(
                folder_path,
                file.replace(
                    ".png",
                    "_mask.png",
                ),
            )

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Missing mask for image: {img_path}"
                )

            image = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE,
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                raise ValueError(
                    f"Could not read image: {img_path}"
                )

            if mask is None:
                raise ValueError(
                    f"Could not read mask: {mask_path}"
                )

            # Preserve binary ground truth.
            mask = (
                mask > 127
            ).astype(np.uint8)

            # Resize the complete ultrasound image.
            image = cv2.resize(
                image,
                (
                    img_width,
                    img_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            # Never interpolate segmentation labels.
            mask = cv2.resize(
                mask,
                (
                    img_width,
                    img_height,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

            image = (
                image.astype(np.float32)
                / 255.0
            )

            mask = mask.astype(
                np.float32
            )

            image = np.expand_dims(
                image,
                axis=-1,
            )

            mask = np.expand_dims(
                mask,
                axis=-1,
            )

            images.append(image)
            masks.append(mask)

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    masks = np.asarray(
        masks,
        dtype=np.float32,
    )

    print(
        f"Total samples loaded: "
        f"{images.shape[0]}"
    )

    return images, masks