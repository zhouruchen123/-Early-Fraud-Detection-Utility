"""
Time-Aware Utility (TAU) for Fraud Detection Model Selection

A novel evaluation metric that incorporates detection timing into
fraud detection model evaluation.
"""

from .tau_metric import (
    TimeAwareUtility,
    TAUConfig,
    evaluate_with_tau,
    calculate_lead_time_advantage
)

__version__ = "1.0.0"
__author__ = "[Author Name]"
__all__ = [
    "TimeAwareUtility",
    "TAUConfig", 
    "evaluate_with_tau",
    "calculate_lead_time_advantage"
]
