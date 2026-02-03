# Time-Aware Utility (TAU) for Fraud Detection Model Selection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **When Early Detection Matters**: A novel model evaluation metric that incorporates detection timing into fraud detection model selection.

## 📖 Overview

Traditional machine learning evaluation metrics (F1 score, AUC) treat all correct detections equally, regardless of when they occur. However, in financial fraud detection, **early detection provides substantially greater value than late detection**.

This repository contains the implementation of **Time-Aware Utility (TAU)**, a novel evaluation metric that:

- Assigns higher utility to earlier fraud detections
- Produces fundamentally different model rankings compared to F1 score (Spearman ρ = -0.414)
- Achieves **100% success rate** across all evaluation criteria
- Enables **30.6 days earlier** fraud detection on average

## 🔬 Key Findings

| Metric | TAU | F1-Optimal | Improvement |
|--------|-----|------------|-------------|
| Average Lead Time | 39.7 days | 70.3 days | **30.6 days earlier** |
| Success Rate | 100% | 0% | - |
| Statistical Significance | p < 0.001 | - | - |

## 📁 Repository Structure

```
├── src/
│   ├── tau_metric.py          # Core TAU implementation
│   ├── models.py              # Model configurations
│   ├── data_loader.py         # Dataset loading utilities
│   └── evaluation.py          # Evaluation functions
├── experiments/
│   ├── run_phase2.py          # Main experiment (TAU vs F1)
│   ├── run_phase3_baselines.py # Baseline comparisons
│   └── run_sensitivity.py     # Sensitivity analysis
├── data/
│   └── README.md              # Dataset download instructions
├── results/
│   └── ...                    # Experiment results
├── plots/
│   └── ...                    # Visualizations
└── docs/
    └── paper.md               # Paper draft
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/time-aware-utility.git
cd time-aware-utility

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.tau_metric import TimeAwareUtility

# Initialize TAU metric
tau = TimeAwareUtility(
    T_max=90,           # Maximum detection window (days)
    k_budget=30,        # Investigation budget (cases)
    days_per_case=2.0   # Days to investigate each case
)

# Calculate TAU score for a model
tau_score = tau.calculate(
    predictions=model_predictions,
    true_labels=y_true,
    detection_times=timestamps
)

# Compare with F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_true, model_predictions)

print(f"TAU Score: {tau_score:.4f}")
print(f"F1 Score: {f1:.4f}")
```

### Running Experiments

```bash
# Run main experiment (Phase 2: TAU vs F1)
python experiments/run_phase2.py --dataset creditcard --seeds 5

# Run baseline comparisons (Phase 3)
python experiments/run_phase3_baselines.py --all-datasets

# Run sensitivity analysis
python experiments/run_sensitivity.py --tmax 30,60,90,120,150,180 --k 10,20,30,50,75,100
```

## 📊 Datasets

This study uses three publicly available fraud detection datasets:

| Dataset | Transactions | Fraud Rate | Source |
|---------|--------------|------------|--------|
| Credit Card Fraud | 284,807 | 0.17% | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| PaySim | 1,000,000 | 0.13% | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| IEEE-CIS | 590,540 | 3.50% | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) |

### Download Instructions

```bash
# Option 1: Using Kaggle API
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
kaggle datasets download -d ealaxi/paysim1 -p data/
kaggle competitions download -c ieee-fraud-detection -p data/

# Option 2: Manual download
# Visit the Kaggle links above and download to data/ directory
```

## 🧮 TAU Formula

The Time-Aware Utility function is defined as:

$$TAU = \sum_{i \in \text{Top-}k} U(t_i, c_i)$$

where:

$$U(t, c) = \max(0, T_{max} - t) \cdot c$$

- $t_i$: Detection time for case $i$
- $c_i$: Confidence score for case $i$
- $T_{max}$: Maximum detection window
- $k$: Investigation budget

## 📈 Results

### Model Ranking Comparison

TAU produces fundamentally different model rankings compared to F1:

- **Spearman correlation**: ρ = -0.414 (inverse relationship)
- **Top-3 model overlap**: 0% (completely different selections)

### Lead Time Advantage

Models selected by TAU detect fraud significantly earlier:

| Dataset | Lead Time Advantage |
|---------|---------------------|
| Credit Card | 43.1 days |
| PaySim | 26.4 days |
| IEEE-CIS | 22.4 days |
| **Average** | **30.6 days** |

### Baseline Comparison

| Method | Success Rate | Lead Time |
|--------|--------------|-----------|
| **Time-Aware Utility** | **100%** | **39.7 days** |
| Cost-Sensitive | 0% | 67.2 days |
| F1-Optimal | 0% | 71.9 days |
| SMOTE | 47% | 51.8 days |
| Class-Balanced | 53% | 57.8 days |
| Profit-Optimal | 0% | 66.1 days |

## 🔧 Configuration

### Default Parameters

```python
DEFAULT_CONFIG = {
    'T_max': 90,              # Maximum detection window (days)
    'k_budget': 30,           # Investigation budget
    'days_per_case': 2.0,     # Days per investigation
    'confidence_delay_factor': 50,  # Confidence-to-delay mapping
    'train_ratio': 0.7        # Train/test split ratio
}
```

### Recommended Parameter Ranges

Based on sensitivity analysis:

- **T_max**: ≥ 90 days (for datasets spanning 180+ days)
- **k_budget**: ≤ 30 cases (TAU advantage increases with smaller budgets)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{tau2025,
  title={Time-Aware Utility for Fraud Detection Model Selection: When Early Detection Matters},
  author={[Author Name]},
  journal={International Review of Financial Analysis},
  year={2025},
  note={Under Review}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue or contact zhouruchen@126.com.

## 🙏 Acknowledgments

- Datasets provided by Kaggle and the machine learning community
- Inspired by the growing need for time-aware evaluation in financial applications
