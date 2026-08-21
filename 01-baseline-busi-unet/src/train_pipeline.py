import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.losses import dice_coefficient

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
    Training pipeline for BASELINE U-Net segmentation.

    - Loss: Binary Crossentropy
    - Metric: Dice coefficient
    - Offline Mean IoU is computed separately
    """

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[dice_coefficient]
    )

    # Default callbacks
    default_callbacks = []

    # Early stopping based on validation Dice
    default_callbacks.append(
        EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    )

    # Reduce LR on Dice plateau
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

    # Merge callbacks
    if callbacks:
        default_callbacks.extend(callbacks)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=default_callbacks,
        verbose=1
    )

    return model, history