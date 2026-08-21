import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_dataset(dataset_path, img_height, img_width, img_channel=1):
    """
    Leakage-free preprocessing for the BUSI dataset.

    Images and masks are loaded independently, resized to the target
    resolution, and normalized. Ground-truth masks are never used to
    determine the image crop or region of interest.
    """
    X_list, y_list = [], []

    for cls in ["benign", "malignant"]:
        cls_path = os.path.join(dataset_path, cls)

        if not os.path.exists(cls_path):
            continue

        img_files = [
            f for f in os.listdir(cls_path)
            if f.endswith(".png") and "_mask" not in f
        ]

        for img_file in img_files:
            img_path = os.path.join(cls_path, img_file)
            mask_path = os.path.join(
                cls_path,
                img_file.replace(".png", "_mask.png")
            )

            if not os.path.exists(mask_path):
                continue

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                continue

            # Binary ground-truth mask.
            mask = (mask > 127).astype(np.uint8)

            # Resize the complete ultrasound image and its mask.
            image = cv2.resize(
                image,
                (img_width, img_height),
                interpolation=cv2.INTER_LINEAR
            )

            mask = cv2.resize(
                mask,
                (img_width, img_height),
                interpolation=cv2.INTER_NEAREST
            )

            # Normalize image to [0, 1].
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)

            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

            X_list.append(image)
            y_list.append(mask)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    print(f"Preprocessed dataset: {X.shape}")

    return X, y


def plot_random_sample(X, y):
    idx = np.random.randint(len(X))

    image = X[idx, :, :, 0]
    mask = y[idx, :, :, 0]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Ultrasound Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground-Truth Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()