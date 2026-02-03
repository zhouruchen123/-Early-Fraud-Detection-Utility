"""
Evaluation Functions for Fraud Detection Experiments

This module provides comprehensive evaluation functions that compare
TAU with traditional metrics like F1 score, and baseline methods.

Author: [Author Name]
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

from .tau_metric import TimeAwareUtility, evaluate_with_tau


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    tau_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with both traditional and TAU metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        tau_config: Configuration for TAU (T_max, k_budget, etc.)
        
    Returns:
        Dictionary of evaluation metrics
    """
    tau_config = tau_config or {"T_max": 90, "k_budget": 30}
    
    # Traditional metrics
    metrics = {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC metrics (require probabilities)
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            metrics["auc_pr"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0
            metrics["auc_pr"] = 0.0
    
    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_positives"] = int(fp)
    metrics["true_negatives"] = int(tn)
    metrics["false_negatives"] = int(fn)
    
    # TAU metrics
    tau_score, tau_details = evaluate_with_tau(
        y_true, y_pred, y_prob,
        T_max=tau_config.get("T_max", 90),
        k_budget=tau_config.get("k_budget", 30)
    )
    
    metrics["tau_score"] = tau_score
    metrics["tau_mean_detection_time"] = tau_details.get("mean_detection_time", 0)
    metrics["tau_cases_detected"] = tau_details.get("n_selected", 0)
    
    return metrics


def compare_rankings(
    f1_scores: Dict[str, float],
    tau_scores: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare model rankings between F1 and TAU metrics.
    
    Args:
        f1_scores: Dictionary mapping model names to F1 scores
        tau_scores: Dictionary mapping model names to TAU scores
        
    Returns:
        Dictionary with ranking comparison statistics
    """
    models = list(f1_scores.keys())
    
    # Create rankings
    f1_ranking = sorted(models, key=lambda m: f1_scores[m], reverse=True)
    tau_ranking = sorted(models, key=lambda m: tau_scores[m], reverse=True)
    
    # Spearman correlation
    f1_ranks = [f1_ranking.index(m) + 1 for m in models]
    tau_ranks = [tau_ranking.index(m) + 1 for m in models]
    
    spearman_corr, spearman_p = stats.spearmanr(f1_ranks, tau_ranks)
    
    # Top-k overlap
    def top_k_overlap(k):
        f1_top_k = set(f1_ranking[:k])
        tau_top_k = set(tau_ranking[:k])
        return len(f1_top_k & tau_top_k) / k
    
    return {
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "f1_ranking": f1_ranking,
        "tau_ranking": tau_ranking,
        "top_1_same": f1_ranking[0] == tau_ranking[0],
        "top_3_overlap": top_k_overlap(3),
        "top_5_overlap": top_k_overlap(5),
        "best_f1_model": f1_ranking[0],
        "best_tau_model": tau_ranking[0]
    }


def calculate_lead_time_statistics(
    tau_detection_times: List[float],
    f1_detection_times: List[float]
) -> Dict[str, float]:
    """
    Calculate lead time advantage statistics.
    
    Args:
        tau_detection_times: Detection times for TAU-optimal models
        f1_detection_times: Detection times for F1-optimal models
        
    Returns:
        Dictionary with lead time statistics
    """
    tau_times = np.array(tau_detection_times)
    f1_times = np.array(f1_detection_times)
    
    # Basic statistics
    tau_mean = np.mean(tau_times)
    f1_mean = np.mean(f1_times)
    advantage = f1_mean - tau_mean
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(tau_times, f1_times)
    
    return {
        "tau_mean_time": tau_mean,
        "tau_std_time": np.std(tau_times),
        "f1_mean_time": f1_mean,
        "f1_std_time": np.std(f1_times),
        "lead_time_advantage": advantage,
        "relative_improvement": advantage / f1_mean if f1_mean > 0 else 0,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05
    }


def evaluate_baseline_methods(
    y_true: np.ndarray,
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    tau_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Evaluate multiple baseline methods and compare with TAU.
    
    Args:
        y_true: True labels
        model_predictions: Dictionary mapping method names to predictions
        tau_config: Configuration for TAU
        
    Returns:
        DataFrame with evaluation results for all methods
    """
    tau_config = tau_config or {"T_max": 90, "k_budget": 30}
    results = []
    
    for method_name, preds in model_predictions.items():
        y_pred = preds["predictions"]
        y_prob = preds.get("probabilities", y_pred.astype(float))
        
        metrics = evaluate_model(y_true, y_pred, y_prob, tau_config)
        metrics["method"] = method_name
        results.append(metrics)
    
    return pd.DataFrame(results)


def success_rate_analysis(
    results_df: pd.DataFrame,
    criteria: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Analyze success rates across different criteria.
    
    Args:
        results_df: DataFrame with evaluation results
        criteria: Dictionary of criteria thresholds
        
    Returns:
        Dictionary with success rate analysis
    """
    criteria = criteria or {
        "f1_threshold": 0.5,
        "tau_threshold": 100,
        "detection_time_threshold": 60
    }
    
    success_counts = {}
    
    for method in results_df["method"].unique():
        method_data = results_df[results_df["method"] == method]
        
        successes = 0
        total_criteria = 0
        
        # F1 criterion
        if "f1" in method_data.columns:
            total_criteria += 1
            if method_data["f1"].mean() >= criteria["f1_threshold"]:
                successes += 1
        
        # TAU criterion
        if "tau_score" in method_data.columns:
            total_criteria += 1
            if method_data["tau_score"].mean() >= criteria["tau_threshold"]:
                successes += 1
        
        # Detection time criterion
        if "tau_mean_detection_time" in method_data.columns:
            total_criteria += 1
            if method_data["tau_mean_detection_time"].mean() <= criteria["detection_time_threshold"]:
                successes += 1
        
        success_counts[method] = {
            "successes": successes,
            "total": total_criteria,
            "rate": successes / total_criteria if total_criteria > 0 else 0
        }
    
    return success_counts


def generate_summary_report(
    evaluation_results: Dict[str, Any],
    output_format: str = "markdown"
) -> str:
    """
    Generate a summary report of evaluation results.
    
    Args:
        evaluation_results: Dictionary containing all evaluation results
        output_format: Output format ('markdown' or 'text')
        
    Returns:
        Formatted summary report string
    """
    if output_format == "markdown":
        report = "# Evaluation Summary Report\n\n"
        
        # Ranking comparison
        if "ranking_comparison" in evaluation_results:
            rc = evaluation_results["ranking_comparison"]
            report += "## Ranking Comparison\n\n"
            report += f"- Spearman Correlation: {rc['spearman_correlation']:.3f}\n"
            report += f"- P-value: {rc['spearman_p_value']:.4f}\n"
            report += f"- Best F1 Model: {rc['best_f1_model']}\n"
            report += f"- Best TAU Model: {rc['best_tau_model']}\n\n"
        
        # Lead time advantage
        if "lead_time" in evaluation_results:
            lt = evaluation_results["lead_time"]
            report += "## Lead Time Advantage\n\n"
            report += f"- TAU Mean Time: {lt['tau_mean_time']:.1f} days\n"
            report += f"- F1 Mean Time: {lt['f1_mean_time']:.1f} days\n"
            report += f"- Advantage: {lt['lead_time_advantage']:.1f} days\n"
            report += f"- Significant: {'Yes' if lt['significant'] else 'No'} (p={lt['p_value']:.4f})\n\n"
        
        return report
    
    else:
        # Plain text format
        report = "EVALUATION SUMMARY REPORT\n"
        report += "=" * 40 + "\n\n"
        # ... similar content in plain text
        return report


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n = 1000
    y_true = np.random.binomial(1, 0.1, n)
    y_pred = np.random.binomial(1, 0.15, n)
    y_prob = np.random.beta(2, 5, n)
    
    # Evaluate
    metrics = evaluate_model(y_true, y_pred, y_prob)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
