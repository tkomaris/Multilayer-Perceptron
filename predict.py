import argparse
import os
import glob
import numpy as np
import pandas as pd
from train import MultilayerPerceptron


def find_latest_model_file(directory="model", prefix="saved_model_"):
    """Find the most recent model file in the specified directory."""
    pattern = os.path.join(directory, f"{prefix}*.npy")
    candidates = sorted(glob.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model files found matching {pattern}")
    return candidates[0]


def load_saved_model(model_path: str) -> MultilayerPerceptron:
    """Load a saved MultilayerPerceptron model from file."""
    data = np.load(model_path, allow_pickle=True).item()
    arch = data["architecture"]
    input_size = arch["input_size"]
    hidden_sizes = arch["hidden_sizes"]
    output_size = arch["output_size"]
    l2_lambda = data.get("l2_lambda", 0.0)

    mlp = MultilayerPerceptron(
        input_size=input_size,
        hidden_layers_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=0.0,
        l2_lambda=l2_lambda,
    )

    for layer, layer_params in zip(mlp.layers, data["layers"]):
        layer.weights = np.array(layer_params["weights"], dtype=float)
        layer.bias = np.array(layer_params["bias"], dtype=float)

    return mlp


def load_test_data(test_file: str):
    """Load test data from CSV file."""
    arr = pd.read_csv(test_file, header=None).values
    y = arr[:, 0].astype(int)
    X = arr[:, 1:]
    return X, y


def calculate_cross_entropy(y_true, probs):
    """Calculate cross-entropy loss."""
    epsilon = 1e-12
    probs_clipped = np.clip(probs, epsilon, 1.0 - epsilon)
    return -np.mean(np.log(probs_clipped[np.arange(y_true.shape[0]), y_true]))


def main():
    parser = argparse.ArgumentParser(description="Run inference using a saved model file.")
    parser.add_argument("model_path", type=str,
                        help="Path to the saved model .npy file (e.g., model/saved_model_YYYYMMDD_HHMMSS.npy)")
    parser.add_argument("--test_file", type=str, default="dataset/data_test.csv",
                        help="Path to the combined test CSV [label, features...]")
    parser.add_argument("test_path", nargs="?", help="Optional positional path to test CSV")
    args = parser.parse_args()

    model = load_saved_model(args.model_path)
    test_file_path = args.test_file if args.test_file else args.test_path or "dataset/data_test.csv"
    X_test, y_true = load_test_data(test_file_path)

    probs = model.forward(X_test, training=False)[-1]
    y_pred = np.argmax(probs, axis=1)

    accuracy = (y_pred == y_true).mean() * 100.0
    print(f"Loaded model: {args.model_path}")
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # Cross-entropy (average negative log-likelihood of true class)
    ce = calculate_cross_entropy(y_true, probs)
    print(f"Test cross-entropy: {ce:.4f}")


if __name__ == "__main__":
    main()

