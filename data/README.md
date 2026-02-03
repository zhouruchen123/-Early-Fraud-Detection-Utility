# Dataset Download Instructions

This directory should contain the three fraud detection datasets used in the experiments.

## Required Datasets

### 1. Credit Card Fraud Detection Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File**: `creditcard.csv`
- **Size**: ~144 MB
- **Transactions**: 284,807
- **Fraud Rate**: 0.17%

**Download Command**:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
unzip data/creditcardfraud.zip -d data/
```

### 2. PaySim Dataset

- **Source**: [Kaggle - PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **File**: `paysim.csv` (rename from `PS_20174392719_1491204439457_log.csv`)
- **Size**: ~470 MB
- **Transactions**: 6,362,620 (we use 1M sample)
- **Fraud Rate**: 0.13%

**Download Command**:
```bash
kaggle datasets download -d ealaxi/paysim1 -p data/
unzip data/paysim1.zip -d data/
mv data/PS_20174392719_1491204439457_log.csv data/paysim.csv
```

### 3. IEEE-CIS Fraud Detection Dataset

- **Source**: [Kaggle - IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **File**: `ieee_cis_train.csv` (merged from train_transaction.csv and train_identity.csv)
- **Size**: ~1.3 GB
- **Transactions**: 590,540
- **Fraud Rate**: 3.50%

**Download Command**:
```bash
kaggle competitions download -c ieee-fraud-detection -p data/
unzip data/ieee-fraud-detection.zip -d data/
# Merge transaction and identity files (see preprocessing script)
```

## Kaggle API Setup

To use the Kaggle API:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → Create New Token
3. Place the downloaded `kaggle.json` in `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Directory Structure After Download

```
data/
├── README.md (this file)
├── creditcard.csv
├── paysim.csv
└── ieee_cis_train.csv
```

## Data Preprocessing

For IEEE-CIS dataset, run the preprocessing script to merge files:

```python
import pandas as pd

# Load transaction and identity data
train_transaction = pd.read_csv('data/train_transaction.csv')
train_identity = pd.read_csv('data/train_identity.csv')

# Merge on TransactionID
train = train_transaction.merge(train_identity, on='TransactionID', how='left')

# Save merged file
train.to_csv('data/ieee_cis_train.csv', index=False)
```

## Citation

If you use these datasets, please cite the original sources:

**Credit Card Fraud**:
> Machine Learning Group - ULB. Credit Card Fraud Detection. Kaggle, 2018.

**PaySim**:
> Lopez-Rojas, E., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. EMSS.

**IEEE-CIS**:
> IEEE Computational Intelligence Society. IEEE-CIS Fraud Detection. Kaggle Competition, 2019.
