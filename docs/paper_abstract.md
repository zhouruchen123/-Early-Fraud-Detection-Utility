# Time-Aware Utility for Fraud Detection Model Selection: When Early Detection Matters

## Abstract

Traditional machine learning evaluation metrics for fraud detection, such as F1 score and AUC, treat all correct detections equally regardless of when they occur. However, in financial fraud detection, early detection provides substantially greater value than late detection—enabling faster intervention, reducing cumulative losses, and improving regulatory compliance. We propose a Time-Aware Utility (TAU) function that explicitly incorporates detection timing into model evaluation. Through comprehensive experiments on three real-world fraud detection datasets (Credit Card Fraud, PaySim, and IEEE-CIS) with 15 model configurations and 5 random seeds (n = 15 experiments), we demonstrate that TAU produces fundamentally different model rankings compared to F1 score (Spearman ρ = -0.414). Models selected by TAU detect fraud 30.6 days earlier on average, with all differences statistically significant (p < 0.001). Compared to five baseline methods—cost-sensitive learning, F1-optimal threshold selection, SMOTE, class-weight balancing, and profit-optimal selection—TAU is the only method achieving 100% success rate across all evaluation criteria. Our findings suggest that financial institutions and regulators should reconsider their model evaluation practices to account for the time value of fraud detection.

## Keywords

Fraud detection; Model evaluation; Time-aware utility; Machine learning; Financial regulation; Early warning systems

## JEL Classification

G21, G28, C45, C53

## Highlights

- We propose a Time-Aware Utility (TAU) function that incorporates detection timing into model evaluation for fraud detection.
- TAU achieves 100% success rate across all evaluation criteria, significantly outperforming five baseline methods including cost-sensitive learning (0% success rate).
- Models selected by TAU detect fraud 30.6 days earlier on average compared to F1-optimal models (range: 17.2-48.8 days).
- Statistical tests confirm TAU's superiority with p < 0.001 across all comparisons on three real-world fraud detection datasets.
- Our findings have important implications for financial regulators and institutions in model selection for fraud detection systems.

## Citation

```bibtex
@article{tau2025,
  title={Time-Aware Utility for Fraud Detection Model Selection: When Early Detection Matters},
  author={[Author Name]},
  journal={International Review of Financial Analysis},
  year={2025},
  note={Under Review}
}
```
