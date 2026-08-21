import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.losses import bce_dice_loss, dice_coefficient


def train_pipeline(
    X_train, y_train,
    X_val, y_val,
    model,
    epochs=50,
    batch_size=8,
    lr=1e-4,
    callbacks=None,
    save_path=None
):
    """
    Training pipeline for U-Net segmentation.

    - Loss: BCE + Dice
    - Metric: Dice coefficient
    - Offline Mean IoU is computed separately
    """

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=bce_dice_loss,
        metrics=[dice_coefficient]
    )

    # Default callbacks if none provided
    default_callbacks = []

    # Early stopping based on val_dice_coefficient
    default_callbacks.append(
        EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",          # maximize Dice
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    )

    # Reduce LR on plateau
    default_callbacks.append(
        ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    )

    # Model checkpoint
    if save_path:
        default_callbacks.append(
            ModelCheckpoint(
                save_path,
                monitor="val_dice_coefficient",
                mode="max",
                save_best_only=True,
                verbose=1
            )
        )

    # Merge with user callbacks if provided
    if callbacks:
        default_callbacks.extend(callbacks)

    # Fit the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=default_callbacks,
        verbose=1
    )

    return model, history