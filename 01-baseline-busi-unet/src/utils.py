import os
import matplotlib.pyplot as plt

def smooth_curve(values, beta=0.9):
    """
    Smooth training curves for better visualization.
    """
    smoothed = []
    v = 0.0
    for i, val in enumerate(values):
        v = beta * v + (1 - beta) * val
        smoothed.append(v / (1 - beta ** (i + 1)))
    return smoothed


def plot_training_loss(history, smooth=True):
    """
    Plot training and validation loss for BASELINE model (BCE only).
    Returns the matplotlib figure object.
    """
    keys = history.history.keys()
    loss_key = "loss" if "loss" in keys else None

    if loss_key is None:
        raise ValueError("Loss key not found in training history.")

    train_loss = history.history[loss_key]
    val_loss = history.history["val_" + loss_key]

    if smooth:
        train_loss = smooth_curve(train_loss)
        val_loss = smooth_curve(val_loss)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss, label="Train BCE")
    ax.plot(val_loss, label="Val BCE")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    plt.show()

    return fig


def plot_validation_metrics(final_dice, final_miou):
    """
    Plot offline evaluation metrics (Dice + Mean IoU) as a bar chart.
    Returns the matplotlib figure object.
    """
    metrics = [final_dice, final_miou]
    names = [
    "Mean Foreground Dice",
    "Mean Foreground IoU",
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(names, metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Model Evaluation Metrics")
    ax.set_ylabel("Score")

    for i, v in enumerate(metrics):
        ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')

    plt.show()
    return fig


def save_figure(fig, fig_folder, filename):
    """
    Save a matplotlib figure to disk.
    """
    os.makedirs(fig_folder, exist_ok=True)
    fig_path = os.path.join(fig_folder, filename)
    fig.savefig(fig_path)
    plt.close(fig)
