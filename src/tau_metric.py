"""
Time-Aware Utility (TAU) Metric for Fraud Detection Model Evaluation

This module implements the TAU metric that incorporates detection timing
into model evaluation for fraud detection systems.

Author: [Author Name]
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TAUConfig:
    """Configuration for Time-Aware Utility calculation."""
    T_max: float = 90.0  # Maximum detection window (days)
    k_budget: int = 30  # Investigation budget (number of cases)
    days_per_case: float = 2.0  # Days required to investigate each case
    confidence_delay_factor: float = 50.0  # Factor for confidence-to-delay mapping
    

class TimeAwareUtility:
    """
    Time-Aware Utility (TAU) metric for fraud detection model evaluation.
    
    TAU assigns higher utility to earlier fraud detections, reflecting the
    operational reality that early detection provides greater value than
    late detection.
    
    The utility function is defined as:
        U(t, c) = max(0, T_max - t) * c
    
    where:
        - t: detection time (days from fraud occurrence)
        - c: confidence score
        - T_max: maximum detection window
    
    Example:
        >>> tau = TimeAwareUtility(T_max=90, k_budget=30)
        >>> score = tau.calculate(predictions, labels, times)
    """
    
    def __init__(
        self,
        T_max: float = 90.0,
        k_budget: int = 30,
        days_per_case: float = 2.0,
        confidence_delay_factor: float = 50.0
    ):
        """
        Initialize TAU metric.
        
        Args:
            T_max: Maximum detection window in days. Detections after T_max
                   have zero utility.
            k_budget: Number of cases that can be investigated (investigation
                      capacity constraint).
            days_per_case: Average days required to investigate each case.
            confidence_delay_factor: Factor for mapping confidence to detection
                                    delay (higher confidence = faster detection).
        """
        self.T_max = T_max
        self.k_budget = k_budget
        self.days_per_case = days_per_case
        self.confidence_delay_factor = confidence_delay_factor
        self.config = TAUConfig(T_max, k_budget, days_per_case, confidence_delay_factor)
    
    def utility_function(self, detection_time: float, confidence: float) -> float:
        """
        Calculate utility for a single detection.
        
        Args:
            detection_time: Time from fraud occurrence to detection (days)
            confidence: Model's confidence score for the prediction
            
        Returns:
            Utility value (higher is better)
        """
        if detection_time >= self.T_max:
            return 0.0
        return max(0, self.T_max - detection_time) * confidence
    
    def estimate_detection_time(
        self,
        confidence: float,
        rank: int,
        base_time: float = 0.0
    ) -> float:
        """
        Estimate detection time based on confidence and investigation queue.
        
        Higher confidence cases are investigated first, leading to faster
        detection. The rank in the queue adds delay based on investigation
        capacity.
        
        Args:
            confidence: Model's confidence score
            rank: Position in investigation queue (0 = first)
            base_time: Base time offset (days)
            
        Returns:
            Estimated detection time in days
        """
        # Queue delay: each case takes days_per_case to investigate
        queue_delay = rank * self.days_per_case
        
        # Confidence-based delay: higher confidence = faster detection
        # This models the intuition that high-confidence cases are easier to verify
        confidence_delay = (1 - confidence) * self.confidence_delay_factor
        
        return base_time + queue_delay + confidence_delay
    
    def calculate(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        confidence_scores: Optional[np.ndarray] = None,
        detection_times: Optional[np.ndarray] = None,
        return_details: bool = False
    ) -> float:
        """
        Calculate TAU score for model predictions.
        
        Args:
            predictions: Binary predictions (0/1)
            true_labels: True labels (0/1)
            confidence_scores: Model confidence scores (if None, uses predictions)
            detection_times: Pre-computed detection times (if None, estimates)
            return_details: If True, return detailed breakdown
            
        Returns:
            TAU score (sum of utilities for top-k true positives)
        """
        predictions = np.asarray(predictions)
        true_labels = np.asarray(true_labels)
        
        if confidence_scores is None:
            confidence_scores = predictions.astype(float)
        else:
            confidence_scores = np.asarray(confidence_scores)
        
        # Find true positives (predicted fraud that is actually fraud)
        tp_mask = (predictions == 1) & (true_labels == 1)
        tp_indices = np.where(tp_mask)[0]
        
        if len(tp_indices) == 0:
            return 0.0 if not return_details else (0.0, {})
        
        # Get confidence scores for true positives
        tp_confidences = confidence_scores[tp_indices]
        
        # Sort by confidence (descending) to prioritize high-confidence cases
        sorted_order = np.argsort(-tp_confidences)
        tp_indices_sorted = tp_indices[sorted_order]
        tp_confidences_sorted = tp_confidences[sorted_order]
        
        # Select top-k cases based on investigation budget
        k = min(self.k_budget, len(tp_indices_sorted))
        top_k_indices = tp_indices_sorted[:k]
        top_k_confidences = tp_confidences_sorted[:k]
        
        # Calculate detection times
        if detection_times is not None:
            top_k_times = detection_times[top_k_indices]
        else:
            top_k_times = np.array([
                self.estimate_detection_time(conf, rank)
                for rank, conf in enumerate(top_k_confidences)
            ])
        
        # Calculate utilities
        utilities = np.array([
            self.utility_function(t, c)
            for t, c in zip(top_k_times, top_k_confidences)
        ])
        
        total_utility = np.sum(utilities)
        
        if return_details:
            details = {
                'n_true_positives': len(tp_indices),
                'n_selected': k,
                'top_k_confidences': top_k_confidences,
                'top_k_times': top_k_times,
                'top_k_utilities': utilities,
                'mean_detection_time': np.mean(top_k_times),
                'mean_utility': np.mean(utilities)
            }
            return total_utility, details
        
        return total_utility
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, np.ndarray]],
        true_labels: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models using TAU metric.
        
        Args:
            model_results: Dictionary mapping model names to their results.
                          Each result should have 'predictions' and 'confidence'.
            true_labels: True labels for all samples.
            
        Returns:
            Dictionary with TAU scores and rankings for each model.
        """
        results = {}
        
        for model_name, model_data in model_results.items():
            predictions = model_data['predictions']
            confidence = model_data.get('confidence', predictions.astype(float))
            
            tau_score, details = self.calculate(
                predictions=predictions,
                true_labels=true_labels,
                confidence_scores=confidence,
                return_details=True
            )
            
            results[model_name] = {
                'tau_score': tau_score,
                'mean_detection_time': details['mean_detection_time'],
                'n_detected': details['n_selected'],
                **details
            }
        
        # Add rankings
        tau_scores = [(name, r['tau_score']) for name, r in results.items()]
        tau_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (name, _) in enumerate(tau_scores, 1):
            results[name]['tau_rank'] = rank
        
        return results


def calculate_lead_time_advantage(
    tau_times: np.ndarray,
    f1_times: np.ndarray
) -> Dict[str, float]:
    """
    Calculate lead time advantage of TAU-optimal vs F1-optimal models.
    
    Args:
        tau_times: Detection times for TAU-optimal model
        f1_times: Detection times for F1-optimal model
        
    Returns:
        Dictionary with lead time statistics
    """
    tau_mean = np.mean(tau_times)
    f1_mean = np.mean(f1_times)
    advantage = f1_mean - tau_mean
    
    return {
        'tau_mean_time': tau_mean,
        'f1_mean_time': f1_mean,
        'lead_time_advantage': advantage,
        'relative_improvement': advantage / f1_mean if f1_mean > 0 else 0
    }


# Convenience function for quick evaluation
def evaluate_with_tau(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    T_max: float = 90.0,
    k_budget: int = 30
) -> Tuple[float, Dict[str, Any]]:
    """
    Quick evaluation using TAU metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        T_max: Maximum detection window
        k_budget: Investigation budget
        
    Returns:
        Tuple of (TAU score, details dictionary)
    """
    tau = TimeAwareUtility(T_max=T_max, k_budget=k_budget)
    return tau.calculate(
        predictions=y_pred,
        true_labels=y_true,
        confidence_scores=y_prob,
        return_details=True
    )


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.1, n_samples)  # 10% fraud rate
    y_prob = np.random.beta(2, 5, n_samples)  # Confidence scores
    y_prob[y_true == 1] += 0.3  # Higher confidence for actual fraud
    y_prob = np.clip(y_prob, 0, 1)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate TAU
    tau = TimeAwareUtility(T_max=90, k_budget=30)
    score, details = tau.calculate(
        predictions=y_pred,
        true_labels=y_true,
        confidence_scores=y_prob,
        return_details=True
    )
    
    print(f"TAU Score: {score:.2f}")
    print(f"Mean Detection Time: {details['mean_detection_time']:.2f} days")
    print(f"Cases Detected: {details['n_selected']}/{details['n_true_positives']}")
