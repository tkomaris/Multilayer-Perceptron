import pandas as pd
import numpy as np
import os
import argparse


class LocalStandardScaler:
    """
    Local implementation of StandardScaler without sklearn.
    Standardizes features by removing the mean and scaling to unit variance.
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        
    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            self: Returns the instance itself
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.scale_ = np.sqrt(self.var_)
        
        # Handle zero variance features
        self.scale_[self.scale_ == 0] = 1.0
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Data to be scaled
            
        Returns:
            X_scaled: Scaled data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This LocalStandardScaler instance is not fitted yet.")
        
        X = np.array(X)
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data
            
        Returns:
            X_scaled: Scaled data
        """
        return self.fit(X).transform(X)


def preprocess_data(file_path):
    """
    Preprocess the dataset by loading, cleaning, and normalizing the data.
    
    Args:
        file_path (str): Path to the CSV dataset file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    df = pd.read_csv(file_path, header=None)

    # Drop the first column (ID) as it's not a feature
    df = df.iloc[:, 1:]

    # Separate features (X) and target (y)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # Convert 'M' and 'B' to 1 and 0 respectively
    y = y.map({'M': 1, 'B': 0})

    # Normalize features using local StandardScaler
    scaler = LocalStandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def split_data(X, y, testing_size, random_state):
    """
    Split the data into training and testing sets with stratification.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        testing_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_val = int(n_samples * testing_size)
    
    # Stratified split by default
    unique_classes = np.unique(y)
    train_indices = []
    val_indices = []
    
    for class_label in unique_classes:
        class_indices = np.where(np.array(y) == class_label)[0]
        np.random.shuffle(class_indices)
        n_class_val = int(len(class_indices) * testing_size)
        val_indices.extend(class_indices[:n_class_val])
        train_indices.extend(class_indices[n_class_val:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    if isinstance(X, np.ndarray):
        X_train, X_val = X[train_indices], X[val_indices]
    else:
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
    
    if isinstance(y, np.ndarray):
        y_train, y_val = y[train_indices], y[val_indices]
    else:
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
    
    return X_train, X_val, y_train, y_val


def save_preprocessed_data(X_train, X_val, y_train, y_val, output_dir='dataset'):
    """
    Save preprocessed data to CSV files with labels and features combined.
    
    Args:
        X_train, X_val: Feature matrices
        y_train, y_val: Target vectors
        output_dir (str): Directory to save the files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Combine labels and features (label as first column)
    train_combined = np.column_stack([y_train, X_train])
    val_combined = np.column_stack([y_val, X_val])
    
    # Save combined datasets
    pd.DataFrame(train_combined).to_csv(f'{output_dir}/data_training.csv', index=False, header=False)
    pd.DataFrame(val_combined).to_csv(f'{output_dir}/data_test.csv', index=False, header=False)
    
    print(f"Preprocessed data saved to {output_dir}/ directory.")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_val.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Files created: data_training.csv, data_test.csv")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for neural network training.")
    parser.add_argument("--input", type=str, default="dataset/data.csv", 
                       help="Path to the input CSV dataset file.")
    parser.add_argument("--output_dir", type=str, default="dataset", 
                       help="Directory to save preprocessed data.")
    parser.add_argument("--testing_size", type=float, default=0.2, 
                       help="Proportion of data for testing.")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Preprocess the data
    X, y = preprocess_data(args.input)
    X_train, X_val, y_train, y_val = split_data(
        X, y, args.testing_size, args.random_state
    )
    
    # Save the preprocessed data
    save_preprocessed_data(X_train, X_val, y_train, y_val, args.output_dir)


if __name__ == '__main__':
    main()

