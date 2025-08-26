import numpy as np
import pandas as pd
import argparse
import os
from datetime import datetime
from preprocessing import split_data


class Perceptron:
    """Single layer perceptron with various activation functions."""
    
    def __init__(self, input_size, output_size, activation_name='relu'):
        """
        Initialize perceptron layer.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output neurons
            activation_name (str): Activation function name
        """
        # Xavier/Glorot initialization for better convergence
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
        self.activation_name = activation_name

    def forward(self, inputs):
        """Forward pass through the perceptron layer."""
        self.inputs = inputs  # Store for backpropagation
        self.z = np.dot(inputs, self.weights) + self.bias
        
        if self.activation_name == 'sigmoid':
            self.output = self.sigmoid(self.z)
        elif self.activation_name == 'relu':
            self.output = self.relu(self.z)
        elif self.activation_name == 'tanh':
            self.output = self.tanh(self.z)
        elif self.activation_name == 'softmax':
            self.output = self.softmax(self.z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        
        return self.output

    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def get_activation_derivative(self, z):
        """Get derivative of activation function."""
        if self.activation_name == 'sigmoid':
            s = self.sigmoid(z)
            return s * (1 - s)
        elif self.activation_name == 'relu':
            return (z > 0).astype(float)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z)**2
        elif self.activation_name == 'softmax':
            # For softmax with cross-entropy, derivative is handled in loss function
            return np.ones_like(z)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")


class Dropout:
    """Dropout layer for regularization."""
    
    def __init__(self, rate):
        """
        Initialize dropout layer.
        
        Args:
            rate (float): Dropout rate (probability of dropping a neuron)
        """
        self.rate = rate
        self.mask = None

    def forward(self, inputs, training=True):
        """Forward pass through dropout layer."""
        if training and self.rate > 0:
            self.mask = np.random.binomial(1, 1.0 - self.rate, size=inputs.shape) / (1.0 - self.rate)
            return inputs * self.mask
        return inputs

    def backward(self, grad):
        """Backward pass through dropout layer."""
        if self.mask is not None:
            return grad * self.mask
        return grad


class AdamOptimizer:
    """Adam optimizer for gradient descent."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate (float): Learning rate
            beta1 (float): Exponential decay rate for first moment estimates
            beta2 (float): Exponential decay rate for second moment estimates
            epsilon (float): Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m_weights = {}  # First moment estimates for weights
        self.v_weights = {}  # Second moment estimates for weights
        self.m_bias = {}     # First moment estimates for bias
        self.v_bias = {}     # Second moment estimates for bias

    def update(self, layer_idx, weights, bias, dw, db):
        """
        Update weights and bias using Adam optimization.
        
        Args:
            layer_idx (int): Layer index
            weights (numpy.ndarray): Current weights
            bias (numpy.ndarray): Current bias
            dw (numpy.ndarray): Weight gradients
            db (numpy.ndarray): Bias gradients
            
        Returns:
            tuple: Updated (weights, bias)
        """
        self.t += 1
        
        # Initialize moment estimates if first time
        if layer_idx not in self.m_weights:
            self.m_weights[layer_idx] = np.zeros_like(weights)
            self.v_weights[layer_idx] = np.zeros_like(weights)
            self.m_bias[layer_idx] = np.zeros_like(bias)
            self.v_bias[layer_idx] = np.zeros_like(bias)
        
        # Update biased first moment estimates
        self.m_weights[layer_idx] = self.beta1 * self.m_weights[layer_idx] + (1 - self.beta1) * dw
        self.m_bias[layer_idx] = self.beta1 * self.m_bias[layer_idx] + (1 - self.beta1) * db
        
        # Update biased second moment estimates
        self.v_weights[layer_idx] = self.beta2 * self.v_weights[layer_idx] + (1 - self.beta2) * (dw ** 2)
        self.v_bias[layer_idx] = self.beta2 * self.v_bias[layer_idx] + (1 - self.beta2) * (db ** 2)
        
        # Compute bias-corrected first moment estimates
        m_weights_corrected = self.m_weights[layer_idx] / (1 - self.beta1 ** self.t)
        m_bias_corrected = self.m_bias[layer_idx] / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second moment estimates
        v_weights_corrected = self.v_weights[layer_idx] / (1 - self.beta2 ** self.t)
        v_bias_corrected = self.v_bias[layer_idx] / (1 - self.beta2 ** self.t)
        
        # Update parameters
        weights_updated = weights - self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        bias_updated = bias - self.learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + self.epsilon)
        
        return weights_updated, bias_updated


class MultilayerPerceptron:
    """Multilayer Perceptron with backpropagation, Adam optimization, and early stopping."""
    
    def __init__(self, input_size, hidden_layers_sizes, output_size, dropout_rate=0.5, l2_lambda=0.01):
        """
        Initialize multilayer perceptron.
        
        Args:
            input_size (int): Number of input features
            hidden_layers_sizes (list): List of hidden layer sizes
            output_size (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
            l2_lambda (float): L2 regularization parameter
        """
        self.layers = []
        self.dropout_layers = []
        self.l2_lambda = l2_lambda
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_layers_sizes + [output_size]
        
        # Hidden layers with ReLU activation
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Perceptron(layer_sizes[i], layer_sizes[i+1], activation_name='relu'))
            self.dropout_layers.append(Dropout(dropout_rate))
        
        # Output layer with softmax activation
        self.layers.append(Perceptron(layer_sizes[-2], layer_sizes[-1], activation_name='softmax'))
        
        # Initialize Adam optimizer
        self.optimizer = None

    def forward(self, inputs, training=True):
        """
        Forward pass through the network.
        
        Args:
            inputs (numpy.ndarray): Input data
            training (bool): Whether in training mode
            
        Returns:
            list: List of activations for each layer
        """
        activations = [inputs]
        current_output = inputs
        
        for i, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
            
            # Apply dropout to hidden layers only
            if i < len(self.dropout_layers):
                current_output = self.dropout_layers[i].forward(current_output, training)
            
            activations.append(current_output)
        
        return activations

    def backward(self, activations, targets):
        """
        Backward pass through the network using backpropagation.
        
        Args:
            activations (list): List of activations from forward pass
            targets (numpy.ndarray): Target labels (one-hot encoded)
        """
        batch_size = targets.shape[0]
        
        # Calculate output layer error (for softmax + cross-entropy)
        output_layer_output = activations[-1]
        error = output_layer_output - targets
        
        # Backpropagate through all layers
        for i in range(len(self.layers) - 1, -1, -1):
            current_layer = self.layers[i]
            current_layer_input = activations[i]
            
            # Apply dropout backward pass for hidden layers
            if i < len(self.dropout_layers):
                error = self.dropout_layers[i].backward(error)
            
            # Calculate gradients
            dw = np.dot(current_layer_input.T, error) / batch_size
            db = np.mean(error, axis=0)
            
            # Add L2 regularization to weight gradients
            dw += self.l2_lambda * current_layer.weights
            
            # Update weights and bias using Adam optimizer
            if self.optimizer is not None:
                current_layer.weights, current_layer.bias = self.optimizer.update(
                    i, current_layer.weights, current_layer.bias, dw, db
                )
            
            # Calculate error for previous layer (except for input layer)
            if i > 0:
                # Get activation derivative for previous layer
                prev_layer = self.layers[i-1]
                activation_derivative = prev_layer.get_activation_derivative(prev_layer.z)
                error = np.dot(error, current_layer.weights.T) * activation_derivative

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, 
              patience, min_delta=0.001):
        """
        Train the neural network with validation data.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Maximum number of epochs
            learning_rate (float): Learning rate for Adam optimizer
            batch_size (int): Batch size for training
            patience (int): Early stopping patience (based on validation loss)
            min_delta (float): Minimum change for early stopping
            
        Returns:
            dict: Training history
        """
        # Initialize Adam optimizer
        self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        
        # Initialize history tracking
        history = {
            "loss": [], "accuracy": [], "precision": [], "recall": [], "f1_score": [],
            "val_loss": [], "val_accuracy": [], "val_precision": [], "val_recall": [], "val_f1_score": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training neural network for {epochs} epochs...")
        print(f"Training on {X_train.shape[0]} samples")
        print(f"Validation on {X_val.shape[0]} samples")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            # Mini-batch training
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                activations = self.forward(X_batch, training=True)
                self.backward(activations, y_batch)
            
            # Evaluate on training set
            train_metrics = self.evaluate_comprehensive(X_train, y_train)
            
            # Evaluate on validation set
            val_metrics = self.evaluate_comprehensive(X_val, y_val)
            
            # Store metrics in history
            for key in train_metrics:
                history[key].append(train_metrics[key])
            for key in val_metrics:
                history[f"val_{key}"].append(val_metrics[key])
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} - "
                  f"Loss: {train_metrics['loss']:.4f} - "
                  f"Acc: {train_metrics['accuracy']:.4f} - "
                  f"Val_Loss: {val_metrics['loss']:.4f} - "
                  f"Val_Acc: {val_metrics['accuracy']:.4f}")
                        
            # Early stopping check based on validation loss
            if val_metrics['loss'] < best_val_loss - min_delta:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
        
        print("-" * 60)
        print("Training completed!")
        return history

    def evaluate_comprehensive(self, X, y):
        """
        Comprehensive evaluation with multiple metrics.
        
        Args:
            X (numpy.ndarray): Input features
            y (numpy.ndarray): Target labels (one-hot encoded)
            
        Returns:
            dict: Dictionary containing various metrics
        """
        activations = self.forward(X, training=False)
        predictions = activations[-1]
        
        # Calculate loss with L2 regularization
        l2_term = 0
        for layer in self.layers:
            l2_term += np.sum(layer.weights**2)
        l2_term *= (self.l2_lambda / (2 * y.shape[0]))
        
        loss = self.categorical_cross_entropy(y, predictions) + l2_term
        
        # Convert to class predictions
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        precision = self.calculate_precision(true_classes, predicted_classes)
        recall = self.calculate_recall(true_classes, predicted_classes)
        f1_score = self.calculate_f1_score(precision, recall)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    def categorical_cross_entropy(self, y_true, y_pred):
        """Calculate categorical cross-entropy loss."""
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def calculate_precision(self, y_true, y_pred):
        """Calculate precision score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def calculate_recall(self, y_true, y_pred):
        """Calculate recall score."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def calculate_f1_score(self, precision, recall):
        """Calculate F1 score."""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            "architecture": {
                "input_size": self.layers[0].weights.shape[0],
                "hidden_sizes": [layer.weights.shape[1] for layer in self.layers[:-1]],
                "output_size": self.layers[-1].weights.shape[1]
            },
            "layers": [
                {
                    "weights": layer.weights.tolist(),
                    "bias": layer.bias.tolist()
                }
                for layer in self.layers
            ],
            "l2_lambda": self.l2_lambda
        }
        
        np.save(filepath, model_data)
        print(f"Model saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train a multilayer perceptron for binary classification.")
    parser.add_argument("--data_file", type=str, default="dataset/data_training.csv",
                        help="Path to the training CSV file [label, features...]")
    parser.add_argument("--layers", type=int, nargs='+', default=[64, 32],
                        help="Hidden layer sizes (e.g., --layers 64 32)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                        help="Dropout rate for regularization")
    parser.add_argument("--l2_lambda", type=float, default=0.01,
                        help="L2 regularization parameter")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="Minimum change for early stopping")
    
    args = parser.parse_args()
    
    # Load training data
    train_data = pd.read_csv(args.data_file, header=None).values
    y_train = train_data[:, 0]  # First column is label
    X_train = train_data[:, 1:]  # Rest are features
    
    X_train, X_val, y_train, y_val = split_data(X_train, y_train, testing_size=0.2, random_state=42)
    
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Label distribution: {np.bincount(y_train.astype(int))}")

    # Convert to one-hot encoding
    y_train_one_hot = np.zeros((y_train.size, 2))
    y_train_one_hot[np.arange(y_train.size), y_train.astype(int)] = 1
    
    y_val_one_hot = np.zeros((y_val.size, 2))
    y_val_one_hot[np.arange(y_val.size), y_val.astype(int)] = 1

    # Initialize and train the model
    input_size = X_train.shape[1]
    output_size = 2

    mlp = MultilayerPerceptron(
        input_size, args.layers, output_size, 
        dropout_rate=args.dropout_rate, l2_lambda=args.l2_lambda
    )
    
    # Train the model
    history = mlp.train(
        X_train, y_train_one_hot, X_val, y_val_one_hot,
        args.epochs, args.learning_rate, args.batch_size, 
        patience=args.patience, min_delta=args.min_delta
    )

    # Create model directory if it doesn't exist
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and history with timestamps
    model_path = os.path.join(model_dir, f"saved_model_{timestamp}.npy")
    history_path = os.path.join(model_dir, f"history_{timestamp}.npy")
    
    mlp.save_model(model_path)
    np.save(history_path, history)
    print(f"Training history saved to {history_path}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_metrics = mlp.evaluate_comprehensive(X_train, y_train_one_hot)
    final_val_metrics = mlp.evaluate_comprehensive(X_val, y_val_one_hot)
    print(f"Final training loss: {final_metrics['loss']:.4f}")
    print(f"Final training accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final validation loss: {final_val_metrics['loss']:.4f}")
    print(f"Final validation accuracy: {final_val_metrics['accuracy']:.4f}")
    

if __name__ == '__main__':
    main()

