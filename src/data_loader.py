"""
Data Loading Utilities for Fraud Detection Experiments

This module provides functions to load and preprocess the three datasets
used in the TAU experiments: Credit Card Fraud, PaySim, and IEEE-CIS.

Author: [Author Name]
License: MIT
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split


# Dataset configurations
DATASET_CONFIGS = {
    "creditcard": {
        "filename": "creditcard.csv",
        "target_col": "Class",
        "time_col": "Time",
        "drop_cols": [],
        "description": "Credit Card Fraud Detection Dataset (Kaggle/ULB)"
    },
    "paysim": {
        "filename": "paysim.csv",
        "target_col": "isFraud",
        "time_col": "step",
        "drop_cols": ["nameOrig", "nameDest", "isFlaggedFraud"],
        "description": "PaySim Mobile Money Simulator Dataset"
    },
    "ieee_cis": {
        "filename": "ieee_cis_train.csv",
        "target_col": "isFraud",
        "time_col": "TransactionDT",
        "drop_cols": ["TransactionID"],
        "description": "IEEE-CIS Fraud Detection Dataset"
    }
}


def load_dataset(
    name: str,
    data_dir: str = "data",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Load a fraud detection dataset.
    
    Args:
        name: Dataset name ('creditcard', 'paysim', 'ieee_cis')
        data_dir: Directory containing the data files
        sample_size: If provided, sample this many rows
        random_state: Random seed for sampling
        
    Returns:
        Tuple of (features DataFrame, labels Series, timestamps Series)
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[name]
    filepath = os.path.join(data_dir, config["filename"])
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            f"Please download the {name} dataset from Kaggle."
        )
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
    
    # Extract target
    y = df[config["target_col"]]
    
    # Extract timestamps if available
    timestamps = None
    if config["time_col"] and config["time_col"] in df.columns:
        timestamps = df[config["time_col"]]
    
    # Prepare features
    drop_cols = config["drop_cols"] + [config["target_col"]]
    if config["time_col"]:
        drop_cols.append(config["time_col"])
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Handle categorical columns (for PaySim and IEEE-CIS)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Fill missing values
    X = X.fillna(0)
    
    return X, y, timestamps


def prepare_temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data temporally (earlier data for training, later for testing).
    
    This is more realistic for fraud detection as it simulates the real-world
    scenario where models are trained on historical data and tested on future data.
    
    Args:
        X: Features DataFrame
        y: Labels Series
        timestamps: Timestamps Series
        train_ratio: Proportion of data for training
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Sort by timestamp
    sorted_idx = timestamps.argsort()
    X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
    y_sorted = y.iloc[sorted_idx].reset_index(drop=True)
    
    # Split at the train_ratio point
    split_idx = int(len(X_sorted) * train_ratio)
    
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test


def prepare_random_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data randomly with optional stratification.
    
    Args:
        X: Features DataFrame
        y: Labels Series
        train_ratio: Proportion of data for training
        random_state: Random seed
        stratify: Whether to stratify by label
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None
    
    return train_test_split(
        X, y,
        train_size=train_ratio,
        random_state=random_state,
        stratify=stratify_param
    )


def get_dataset_info(name: str, data_dir: str = "data") -> Dict[str, Any]:
    """
    Get information about a dataset without loading all data.
    
    Args:
        name: Dataset name
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with dataset information
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}")
    
    config = DATASET_CONFIGS[name]
    filepath = os.path.join(data_dir, config["filename"])
    
    info = {
        "name": name,
        "description": config["description"],
        "filepath": filepath,
        "exists": os.path.exists(filepath)
    }
    
    if info["exists"]:
        # Read just first few rows to get column info
        df_sample = pd.read_csv(filepath, nrows=1000)
        info["n_rows_sample"] = len(df_sample)
        info["n_columns"] = len(df_sample.columns)
        info["columns"] = list(df_sample.columns)
        info["fraud_rate_sample"] = df_sample[config["target_col"]].mean()
    
    return info


def generate_synthetic_timestamps(
    n_samples: int,
    time_span_days: int = 180,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic timestamps for datasets without time information.
    
    Args:
        n_samples: Number of samples
        time_span_days: Total time span in days
        random_state: Random seed
        
    Returns:
        Array of timestamps (in days from start)
    """
    np.random.seed(random_state)
    return np.sort(np.random.uniform(0, time_span_days, n_samples))


if __name__ == "__main__":
    # Print dataset information
    print("Available Datasets:")
    print("-" * 60)
    for name in DATASET_CONFIGS:
        config = DATASET_CONFIGS[name]
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Target column: {config['target_col']}")
        print(f"  Time column: {config['time_col']}")
