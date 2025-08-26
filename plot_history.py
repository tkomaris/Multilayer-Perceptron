import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Plot training history.")
    parser.add_argument("history_path", type=str, help="Path to history file under ./model/, e.g., model/history_YYYYMMDD_HHMMSS.npy")
    args = parser.parse_args()

    history_path = args.history_path
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")
    base = os.path.basename(history_path)
    parent = os.path.basename(os.path.dirname(history_path))
    if parent != "model" or not (base.startswith("history_") and base.endswith(".npy")):
        raise ValueError("History path must be in ./model/ and match 'history_*.npy'.")

    history = np.load(history_path, allow_pickle=True).item()
    os.makedirs("model", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("model", f"loss_plot_{timestamp}.png"))
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("model", f"accuracy_plot_{timestamp}.png"))
    plt.show()

if __name__ == '__main__':
    main()


