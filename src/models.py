"""
Model Configurations for Fraud Detection Experiments

This module defines the model configurations used in the TAU experiments,
including Logistic Regression, Random Forest, and Gradient Boosting variants.

Author: [Author Name]
License: MIT
"""

from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Model configuration definitions
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Logistic Regression variants
    "LR-Baseline": {
        "model_class": LogisticRegression,
        "params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42
        },
        "description": "Standard logistic regression"
    },
    "LR-Balanced": {
        "model_class": LogisticRegression,
        "params": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "description": "Logistic regression with balanced class weights"
    },
    "LR-Aggressive": {
        "model_class": LogisticRegression,
        "params": {
            "C": 0.1,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "description": "Logistic regression with stronger regularization"
    },
    "LR-L1": {
        "model_class": LogisticRegression,
        "params": {
            "C": 1.0,
            "penalty": "l1",
            "solver": "saga",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "description": "Logistic regression with L1 regularization"
    },
    "LR-ElasticNet": {
        "model_class": LogisticRegression,
        "params": {
            "C": 1.0,
            "penalty": "elasticnet",
            "solver": "saga",
            "l1_ratio": 0.5,
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": 42
        },
        "description": "Logistic regression with ElasticNet regularization"
    },
    
    # Random Forest variants
    "RF-Shallow": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "description": "Shallow random forest (max_depth=5)"
    },
    "RF-Medium": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "description": "Medium depth random forest (max_depth=10)"
    },
    "RF-Deep": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 20,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "description": "Deep random forest (max_depth=20)"
    },
    "RF-Large": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 15,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "description": "Large random forest (200 trees)"
    },
    "RF-Unlimited": {
        "model_class": RandomForestClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": None,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        },
        "description": "Random forest with unlimited depth"
    },
    
    # Gradient Boosting variants
    "GB-Standard": {
        "model_class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "description": "Standard gradient boosting"
    },
    "GB-Deep": {
        "model_class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "description": "Deeper gradient boosting (max_depth=5)"
    },
    "GB-Aggressive": {
        "model_class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.2,
            "random_state": 42
        },
        "description": "Aggressive gradient boosting (higher learning rate)"
    },
    "GB-Conservative": {
        "model_class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "random_state": 42
        },
        "description": "Conservative gradient boosting (lower learning rate, more trees)"
    },
    "GB-Large": {
        "model_class": GradientBoostingClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "description": "Large gradient boosting (200 trees)"
    }
}


def get_model(name: str, random_state: int = 42) -> Any:
    """
    Get a model instance by name.
    
    Args:
        name: Model configuration name
        random_state: Random seed for reproducibility
        
    Returns:
        Configured model instance
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[name]
    params = config["params"].copy()
    
    # Update random state if applicable
    if "random_state" in params:
        params["random_state"] = random_state
    
    return config["model_class"](**params)


def get_model_with_scaler(name: str, random_state: int = 42) -> Pipeline:
    """
    Get a model pipeline with StandardScaler preprocessing.
    
    Args:
        name: Model configuration name
        random_state: Random seed for reproducibility
        
    Returns:
        Pipeline with scaler and model
    """
    model = get_model(name, random_state)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def get_all_model_names() -> List[str]:
    """Get list of all available model names."""
    return list(MODEL_CONFIGS.keys())


def get_model_description(name: str) -> str:
    """Get description for a model configuration."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_CONFIGS[name]["description"]


# Default model subset for quick experiments
DEFAULT_MODELS = [
    "LR-Baseline",
    "LR-Balanced",
    "RF-Medium",
    "RF-Deep",
    "GB-Standard",
    "GB-Aggressive"
]


if __name__ == "__main__":
    # Print all available models
    print("Available Model Configurations:")
    print("-" * 60)
    for name in get_all_model_names():
        desc = get_model_description(name)
        print(f"  {name}: {desc}")
