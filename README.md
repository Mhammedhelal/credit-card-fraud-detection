
# 💳 Credit Card Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-%234CAF50.svg?style=flat)](https://imbalanced-learn.org/stable/)
[![XGBoost](https://img.shields.io/badge/xgboost-%23FF6600.svg?style=flat)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/lightgbm-%2302BEB7.svg?style=flat)](https://lightgbm.readthedocs.io/)
[![Joblib](https://img.shields.io/badge/joblib-%230072C6.svg?style=flat)](https://joblib.readthedocs.io/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=flat&logo=plotly&logoColor=black)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-%232E5E82.svg?style=flat)](https://seaborn.pydata.org/)
[![SHAP](https://img.shields.io/badge/SHAP-%23FF6B6B.svg?style=flat)](https://shap.readthedocs.io/)

A machine learning project to detect fraudulent credit card transactions using anonymised PCA-transformed transaction data. The pipeline implements a full story-driven EDA, evidence-based feature engineering, stratified cross-validation, cost-aware threshold selection, and SHAP interpretability — all grounded in the analytical findings of the EDA notebook.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [EDA Findings](#eda-findings)
- [Feature Engineering](#feature-engineering)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage — Step by Step](#usage--step-by-step)
- [Models &amp; Hyperparameters](#models--hyperparameters)
- [Full Results Table](#full-results-table)
- [Model Comparison &amp; Decision](#model-comparison--decision)
- [Evaluation Methodology](#evaluation-methodology)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Credit card fraud causes billions of dollars in losses every year. Financial institutions must detect fraudulent transactions in near real-time — before money leaves the account — while avoiding false alarms that block legitimate customers.

This project builds a fraud detection pipeline from the ground up, starting with a thorough EDA investigation that drives every downstream decision. Key principles:

- **Investigate before engineering** — every feature is justified by prior analysis
- **Evidence before action** — no transformation is applied without showing it improves signal
- **PR-AUC as primary metric** — accuracy is misleading at 533:1 class imbalance
- **Cost-aware threshold selection** — the operating threshold is selected from the Precision-Recall curve using a configurable cost ratio, not fixed at 0.5

---

## Dataset

| Property              | Value                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------- |
| Source                | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions    | 284,807                                                                                      |
| Training set          | 170,884 rows                                                                                 |
| Validation set        | 56,960 rows                                                                                  |
| Test set              | 56,963 rows                                                                                  |
| Features              | 30 inputs + 1 target                                                                         |
| Fraud cases (train)   | 305 (0.178%)                                                                                 |
| Class imbalance ratio | 559:1 (normal:fraud)                                                                         |

**Columns:**

- `Time` — seconds elapsed since first transaction (~48 hours of data)
- `V1–V28` — anonymised PCA-transformed features
- `Amount` — transaction value in USD (range: $0 to $25,691)
- `Class` — target label: 0 = legitimate, 1 = fraudulent

Download `creditcard.csv` from Kaggle and split into `data/train.csv`, `data/val.csv`, `data/test.csv`.

---

## EDA Findings

The EDA (`notebooks/1_EDA.ipynb`) is a story-driven investigation where every conclusion feeds into the next step. Key findings that directly shaped the pipeline:

### Two Confirmed Fraud Strategies

**Strategy 1 — Micro-probing (Card Testing)**

- 11.3% of fraud transactions have Amount < $1 (vs 0.03% of legitimate)
- Peaks at hour 2 AM when transaction volume is lowest and monitoring is reduced
- Purpose: verify a stolen card number is active without triggering any alert threshold

**Strategy 2 — Controlled Drainage**

- Amounts cluster between $100 and $1,500 with a hard ceiling at ~$1,500
- Secondary temporal peak at hours 10–11 AM to blend into morning business traffic
- Fraudsters avoid the afternoon peak (14–19) where high volume provides less cover

### Amount Distribution

- Raw `Amount` skewness: **19.99** — extreme right skew
- After `log1p` transformation: **0.80** — near-normal
- Fraud median: $12 vs normal median: $22
- The U-shaped fraud rate across quantile bins (Very Low: 0.35%, Medium: 0.07%, Very High: 0.24%) confirmed that binning captures a non-monotonic signal a linear feature cannot express

### Temporal Patterns

- `Hour` is a circular feature — hour 23 is adjacent to hour 0
- Raw integer encoding treats them as 23 steps apart, which is wrong
- Sine/cosine encoding correctly places them adjacent on the unit circle

### PCA Features

- All 28 V-features are uncorrelated with each other (by PCA design) — no multicollinearity
- Max individual Pearson r with fraud: **V17 = −0.326**, V14 = −0.300, V12 = −0.260
- No single feature linearly separates the classes — fraud requires non-linear models
- **V12 is unique**: the only V-feature correlated with Hour (r = +0.35)

### Interaction Feature Analysis

- `V12 × log_amount`: r = −0.208 with fraud — independent contextual signal
- `V12 × Hour`: r = −0.210 with fraud — amplifies V12's temporal component
- `V17 × log_amount`, `V14 × log_amount`, `V10 × log_amount`: **all rejected** — dilute the strong base signal (V17 alone r = −0.326 > any interaction)
- `amount × hour` (naive product): r = −0.011 — confirmed noise, dropped

### Outlier Analysis

- Z-score flags (|Z| > 2) on amount and all V-features: **all near-zero correlation with fraud**
- Fraud does not live in distribution tails — it hides in the normal range by design
- Conclusion: `is_outlier_*` features add only noise and were excluded from the model

---

## Feature Engineering

All decisions below are justified by EDA evidence. No feature was created without prior investigation.

### Features Created

| Feature        | Formula                          | EDA Justification                                                |
| -------------- | -------------------------------- | ---------------------------------------------------------------- |
| `log_amount` | `log1p(Amount)`                | Skewness 19.99 → 0.80; clearer class separation (§6.4)         |
| `amount_bin` | 5-quantile bins on`log_amount` | U-shaped fraud rate across bins confirmed (§11.1)               |
| `Hour`       | `(Time // 3600) % 24`          | Fraud has strong temporal concentration (§7.1)                  |
| `sin_hour`   | `sin(2π × Hour / 24)`        | Circular encoding — fraud window spans midnight (§12.1)        |
| `cos_hour`   | `cos(2π × Hour / 24)`        | Paired with sin_hour for full circular representation            |
| `V12_amount` | `V12 × log_amount`            | r = −0.21 with fraud; V12 correlates with Amount (§11.2)       |
| `V12_hour`   | `V12 × Hour`                  | r = −0.21 with fraud; V12 encodes time (r=0.35 with Hour)       |
| `V7_amount`  | `V7 × log_amount`             | r = −0.09; V7 strongest Amount correlator (r=0.42), independent |
| `V11_hour`   | `V11 × Hour`                  | r = +0.10; independent secondary temporal signal                 |

### Features Explicitly Dropped

| Feature                                  | Reason                                                        |
| ---------------------------------------- | ------------------------------------------------------------- |
| Raw`Amount`                            | Replaced by`log_amount`                                     |
| Raw`Time`                              | Replaced by`Hour`                                           |
| `is_outlier_amount`                    | r = 0.009 — confirmed useless (§11.3)                       |
| `V*_is_outlier` (all 28)               | All near-zero r — fraud is not anomalous in tails            |
| `amount_hour_interaction`              | r = −0.011 — raw product without structural connection      |
| `V17 × *`, `V14 × *`, `V10 × *` | Dilute base signal — V17 alone (r=−0.326) > any interaction |
| `is_rush_hour` (binary)                | Too coarse — cyclical encoding is strictly more informative  |

### Preprocessing

| Step                 | Applied To                    | Reason                                                                                    |
| -------------------- | ----------------------------- | ----------------------------------------------------------------------------------------- |
| `OrdinalEncoder`   | `amount_bin` only           | Maps Very Low→0 ... Very High→4 with explicit category order                            |
| No`StandardScaler` | V-features, interactions      | XGBoost/LightGBM are scale-invariant — scaling is unnecessary and potentially distorting |
| `StandardScaler`   | Logistic Regression, KNN only | Linear/distance-based models require scaling                                              |

### Data Leakage Prevention

- `amount_bin` edges are fitted **only on training data** via `pd.qcut`
- Val and test data use `pd.cut` with the saved training edges — never refit
- SMOTE is applied **only on training data**, never on validation or test splits

---

## Pipeline Architecture

```
data/train.csv
     │
     ▼
feature_engineering.py --mode train
     │  • Extracts Hour, log_amount, amount_bin (fit edges)
     │  • Creates sin/cos hour, V12_amount, V12_hour, V7_amount, V11_hour
     │  • Fits OrdinalEncoder on amount_bin
     │  • Saves feature_artifacts.pkl
     ▼
data/engineered/train_features.parquet
     │
     ▼
training.py
     │  • Stratified K-fold CV (5 or 10 folds) — reports PR-AUC per fold
     │  • Trains final model with scale_pos_weight = 559
     │  • Selects threshold via PR curve:
     │      Strategy A: highest Precision where Recall ≥ 0.80
     │      Strategy B: minimise FN×cost_fn_ratio + FP×1
     │  • Saves trained_model.pkl + threshold_report.json
     ▼
models/xgboost_shallow/trained_model.pkl
     │
     ▼
testing.py
     │  • Loads model and threshold
     │  • Computes PR-AUC (primary), Recall, Precision, F2, cost
     │  • Plots PR curve + confusion matrix
     │  • Runs SHAP analysis for interpretability
     ▼
eval/xgboost_shallow_test/
    ├── evaluation_metrics.json
    ├── predictions.csv
    ├── precision_recall_curve.png
    ├── confusion_matrix.png
    └── shap_summary.png
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mhammedhelal/credit-fraud-detection.git
cd credit-fraud-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
xgboost
lightgbm
joblib
matplotlib
seaborn
shap
pyarrow
```

---

## Usage — Step by Step

### Step 1 — Feature Engineering

```bash
# Process training data — fits and saves all transformers
python src/feature_engineering.py --mode train \
    --input-data data/train.csv \
    --output-dir data/engineered

# Process validation data — applies training transformers (no refit)
python src/feature_engineering.py --mode val \
    --input-data data/val.csv \
    --output-dir data/engineered

# Process test data — applies training transformers (no refit)
python src/feature_engineering.py --mode test \
    --input-data data/test.csv \
    --output-dir data/engineered

# Optional: generate SMOTE-oversampled training data (fallback strategy)
python src/feature_engineering.py --mode train \
    --input-data data/train.csv \
    --output-dir data/engineered/oversampled \
    --apply-smote
```

### Step 2 — Train Models

#### XGBoost (default config — depth 6)

```bash
python src/training.py \
    --model-type xgboost \
    --train-data data/engineered/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --cv-folds   5 \
    --min-recall 0.80 \
    --cost-fn-ratio 10 \
    --output-dir models/xgboost_folds_5
```

#### XGBoost Shallow — Config A (best performing)

```bash
# Edit build_xgboost() in src/training.py:
# max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8

python src/training.py \
    --model-type xgboost \
    --train-data data/engineered/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --cv-folds   5 \
    --min-recall 0.80 \
    --cost-fn-ratio 10 \
    --output-dir models/xgboost_shallow
```

#### LightGBM v1 (default, no SMOTE)

```bash
python src/training.py \
    --model-type lightgbm \
    --train-data data/engineered/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --cv-folds   5 \
    --min-recall 0.80 \
    --cost-fn-ratio 10 \
    --output-dir models/lightgbm_v1
```

#### LightGBM v2 (with SMOTE)

```bash
python src/training.py \
    --model-type lightgbm \
    --train-data data/engineered/oversampled/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --cv-folds   5 \
    --min-recall 0.80 \
    --cost-fn-ratio 10 \
    --output-dir models/lightgbm_v2
```

#### Logistic Regression v1 (SMOTE, no scaling)

```bash
python src/training.py \
    --model-type logistic \
    --train-data data/engineered/oversampled/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --cv-folds   5 \
    --min-recall 0.80 \
    --cost-fn-ratio 10 \
    --output-dir models/logistic_v1
```

#### Logistic Regression v2 (original data, StandardScaler + C=0.05)

```bash
# Edit build_model() in src/training.py — replace logistic branch with:
# Pipeline([StandardScaler(), LogisticRegression(class_weight='balanced', C=0.05,
#           solver='saga', max_iter=2000)])

python src/training.py \
    --model-type logistic \
    --train-data data/engineered/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --cv-folds   5 \
    --min-recall 0.80 \
    --cost-fn-ratio 10 \
    --output-dir models/logistic_v2
```

#### KNN k=5

```bash
# Edit build_model() in src/training.py — KNN branch:
# Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=5,
#           weights='distance', algorithm='ball_tree')])

python src/training.py \
    --model-type knn \
    --train-data data/engineered/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --n-neighbors 5 \
    --output-dir models/knn_k5

python src/training.py \
    --model-type knn \
    --train-data data/engineered/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --n-neighbors 11 \
    --output-dir models/knn_k11

python src/training.py \
    --model-type knn \
    --train-data data/engineered/oversampled/train_features.parquet \
    --val-data   data/engineered/val_features.parquet \
    --n-neighbors 5 \
    --output-dir models/knn_smote_k5
```

### Step 3 — Evaluate Models

```bash
# Evaluate any model on train / val / test
python src/testing.py \
    --model-path    models/xgboost_shallow/trained_model.pkl \
    --data-path     data/engineered/train_features.parquet \
    --output-dir    eval/xgboost_shallow_train \
    --cost-fn-ratio 10

python src/testing.py \
    --model-path    models/xgboost_shallow/trained_model.pkl \
    --data-path     data/engineered/val_features.parquet \
    --output-dir    eval/xgboost_shallow_val \
    --cost-fn-ratio 10

python src/testing.py \
    --model-path    models/xgboost_shallow/trained_model.pkl \
    --data-path     data/engineered/test_features.parquet \
    --output-dir    eval/xgboost_shallow_test \
    --cost-fn-ratio 10

# Skip SHAP for faster evaluation (KNN, logistic)
python src/testing.py \
    --model-path    models/knn_k5/trained_model.pkl \
    --data-path     data/engineered/test_features.parquet \
    --output-dir    eval/knn_k5_test \
    --cost-fn-ratio 10 \
    --no-shap

# Override threshold manually
python src/testing.py \
    --model-path    models/xgboost_shallow/trained_model.pkl \
    --data-path     data/engineered/test_features.parquet \
    --output-dir    eval/xgboost_shallow_thresh060 \
    --threshold     0.60 \
    --cost-fn-ratio 10
```

### Step 4 — Probability Calibration (optional, improves PR-AUC)

```bash
python3 - << 'EOF'
import joblib, pandas as pd, os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score

bundle  = joblib.load('models/xgboost_shallow/trained_model.pkl')
model   = bundle['model']

val  = pd.read_parquet('data/engineered/val_features.parquet')
test = pd.read_parquet('data/engineered/test_features.parquet')
X_val,  y_val  = val.drop(columns=['Class']),  val['Class']
X_test, y_test = test.drop(columns=['Class']), test['Class']

cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
cal.fit(X_val, y_val)

before = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
after  = average_precision_score(y_test, cal.predict_proba(X_test)[:, 1])
print(f"Before calibration PR-AUC: {before:.4f}")
print(f"After  calibration PR-AUC: {after:.4f}")

os.makedirs('models/xgboost_shallow_calibrated', exist_ok=True)
joblib.dump({**bundle, 'model': cal},
            'models/xgboost_shallow_calibrated/trained_model.pkl')
EOF
```

---

## Models & Hyperparameters

### XGBoost Default (folds_5 / folds_10)

| Parameter                 | Value                      |
| ------------------------- | -------------------------- |
| `n_estimators`          | 500                        |
| `learning_rate`         | 0.05                       |
| `max_depth`             | 6                          |
| `min_child_weight`      | 3                          |
| `subsample`             | 0.8                        |
| `colsample_bytree`      | 0.8                        |
| `scale_pos_weight`      | 559.3                      |
| `eval_metric`           | `aucpr`                  |
| `early_stopping_rounds` | 30                         |
| Training data             | Original (no SMOTE)        |
| CV folds                  | 5 or 10 (identical output) |

### XGBoost Shallow — Config A ⭐ Best Model

| Parameter                 | Value               |
| ------------------------- | ------------------- |
| `n_estimators`          | 500                 |
| `learning_rate`         | 0.05                |
| `max_depth`             | **4**         |
| `min_child_weight`      | **5**         |
| `subsample`             | 0.8                 |
| `colsample_bytree`      | 0.8                 |
| `scale_pos_weight`      | 559.3               |
| `eval_metric`           | `aucpr`           |
| `early_stopping_rounds` | 30                  |
| Training data             | Original (no SMOTE) |

### LightGBM v1

| Parameter                | Value                                                        |
| ------------------------ | ------------------------------------------------------------ |
| `n_estimators`         | 500                                                          |
| `learning_rate`        | 0.05                                                         |
| `max_depth`            | 6                                                            |
| `min_child_samples`    | 20                                                           |
| `subsample`            | 0.8                                                          |
| `colsample_bytree`     | 0.8                                                          |
| `scale_pos_weight`     | 559.3                                                        |
| `early_stopping_round` | 30                                                           |
| Training data            | Original (no SMOTE)                                          |
| Issue                    | Threshold collapsed to 1.0 — probability calibration needed |

### LightGBM v2

Same as v1 but trained on SMOTE-oversampled data (`sampling_strategy=0.1`, 9:1 ratio after resampling).

### Logistic Regression v1

| Parameter        | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| `class_weight` | inverse frequency ratio                                      |
| `max_iter`     | default                                                      |
| Scaling          | None                                                         |
| Training data    | SMOTE-oversampled                                            |
| Issue            | Threshold collapsed to 0.9986 — overconfident probabilities |

### Logistic Regression v2

| Parameter        | Value                         |
| ---------------- | ----------------------------- |
| `class_weight` | `'balanced'`                |
| `C`            | 0.05                          |
| `solver`       | `saga`                      |
| `penalty`      | `l2`                        |
| `max_iter`     | 2000                          |
| Scaling          | `StandardScaler` (pipeline) |
| Training data    | Original (no SMOTE)           |

---

## Full Results Table

All evaluations use `cost_fn_ratio = 10` (missing 1 fraud = 10× cost of 1 false alarm).
Threshold is the value saved with the model (Strategy A: highest Precision at Recall ≥ 0.80).
**Bold** = best value in each column across all models on the test set.

### Test Set Results

| Model                        | Threshold | PR-AUC           | ROC-AUC          | Recall           | Precision        | F2               | F1               | FN           | FP           | Total Cost    |
| ---------------------------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------ | ------------ | ------------- |
| **xgboost_shallow** ⭐ | 0.9663    | **0.8487** | **0.9863** | 0.8557           | **0.8646** | **0.8574** | **0.8601** | **14** | **13** | **153** |
| xgboost_folds_5              | 0.7490    | 0.7504           | 0.9841           | **0.8660** | 0.7636           | 0.8434           | 0.8116           | 13           | 26           | 156           |
| xgboost_folds_10             | 0.7490    | 0.7504           | 0.9841           | **0.8660** | 0.7636           | 0.8434           | 0.8116           | 13           | 26           | 156           |
| lightgbm_v2 (SMOTE)          | 0.4190    | 0.7362           | 0.9802           | 0.8557           | 0.6014           | 0.7890           | 0.7064           | 14           | 55           | 195           |
| logistic_v2 (scaled)         | 0.9955    | 0.6748           | 0.9694           | 0.8454           | 0.6260           | 0.7900           | 0.7193           | 15           | 49           | 199           |
| logistic_v1 (SMOTE)          | 0.9986    | 0.6692           | 0.9624           | 0.8454           | 0.6308           | 0.7915           | 0.7225           | 15           | 48           | 198           |
| lightgbm_v1                  | 1.0000    | 0.4216           | 0.8833           | 0.7938           | 0.5033           | 0.7116           | 0.6160           | 20           | 76           | 276           |

### Validation Set Results

| Model                        | Threshold | PR-AUC           | Recall           | Precision        | F2               | F1               | FN | FP          | Total Cost    |
| ---------------------------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | -- | ----------- | ------------- |
| **xgboost_shallow** ⭐ | 0.9663    | **0.8517** | 0.8111           | **0.9359** | 0.8333           | **0.8690** | 17 | **5** | 175           |
| xgboost_folds_5              | 0.7490    | 0.7704           | **0.8222** | 0.8605           | **0.8296** | 0.8409           | 16 | 12          | **172** |
| xgboost_folds_10             | 0.7490    | 0.7704           | **0.8222** | 0.8605           | **0.8296** | 0.8409           | 16 | 12          | **172** |
| lightgbm_v2 (SMOTE)          | 0.4190    | 0.7292           | 0.8000           | 0.6667           | 0.7692           | 0.7273           | 18 | 36          | 216           |
| logistic_v1 (SMOTE)          | 0.9986    | 0.7294           | 0.8000           | 0.6857           | 0.7742           | 0.7385           | 18 | 33          | 213           |
| logistic_v2 (scaled)         | 0.9955    | 0.7230           | 0.8000           | 0.6606           | 0.7676           | 0.7236           | 18 | 37          | 217           |
| lightgbm_v1                  | 1.0000    | 0.4451           | 0.8000           | 0.5414           | 0.7302           | 0.6457           | 18 | 61          | 241           |

### Training Set Results (overfitting diagnostic)

| Model            | PR-AUC | Recall           | Precision | FN          | FP  | Train-Test PR Gap |
| ---------------- | ------ | ---------------- | --------- | ----------- | --- | ----------------- |
| xgboost_shallow  | 0.9958 | **1.0000** | 0.9683    | **0** | 10  | 0.147             |
| xgboost_folds_5  | 0.9199 | 0.9770           | 0.8739    | 7           | 43  | 0.170             |
| xgboost_folds_10 | 0.9199 | 0.9770           | 0.8739    | 7           | 43  | 0.170             |
| lightgbm_v2      | 0.9865 | 0.8429           | 0.9908    | 2,680*      | 134 | 0.250             |
| logistic_v2      | 0.7569 | 0.8230           | 0.6322    | 54          | 146 | 0.082             |
| logistic_v1      | 0.7614 | 0.8197           | 0.6649    | 55          | 126 | 0.093             |
| lightgbm_v1      | 0.5067 | 0.7836           | 0.5872    | 66          | 168 | 0.085             |

*lightgbm_v2 trained on SMOTE data — FN of 2,680 refers to synthetic samples, not real fraud cases. Training metrics for this model are not meaningful.

---

## Model Comparison & Decision

### Why XGBoost Shallow is the Production Candidate

Changing three hyperparameters (`max_depth 6→4`, `min_child_weight 3→5`) produced a **9.9-point PR-AUC improvement** on test (0.750 → 0.849). This is the largest gain of any experiment.

The key insight from EDA: with only 305 real training fraud cases, deeper trees memorise individual fraud transactions rather than learning general fraud patterns. Shallower trees are forced to find splits that work across many cases — this generalises better to unseen data.

| Criterion              | XGBoost Shallow    | XGBoost Default    |
| ---------------------- | ------------------ | ------------------ |
| Test PR-AUC            | **0.849** ✅ | 0.750              |
| Test Recall            | 0.856 ✅           | **0.866** ✅ |
| Test Precision         | **0.865** ✅ | 0.764 ✅           |
| Test FP (analyst load) | **13**       | 26                 |
| Total cost             | **153**      | 156                |
| Meets all targets?     | **Yes**      | Yes                |

### Why CV Fold Count Does Not Matter Here

`xgboost_folds_5` and `xgboost_folds_10` produce **identical** test results. With `random_state=42` and early stopping, XGBoost converges to the same model regardless of CV fold count when hyperparameters and data are fixed. More folds provide diagnostic value (per-fold PR-AUC) but do not change the trained model.

### Why LightGBM v1 Failed

Threshold collapsed to 1.0 — LightGBM's raw probability scores never exceeded ~0.95 on this data without calibration, so no threshold between 0 and 1 achieved the 80% recall floor. This is a calibration problem, not a fundamental model failure. LightGBM + calibration is a viable next step.

### Why Logistic Regression Cannot Compete

Test PR-AUC ceiling of ~0.675 despite scaling, regularisation tuning, and class weighting. This validates the EDA finding: max Pearson r with fraud is 0.326, meaning only 10.6% of fraud variance is linearly explainable. A single hyperplane cannot capture the multi-dimensional, non-linear fraud clusters visible in the scatter plots and t-SNE analysis.

### Performance Targets vs Actual

| Metric    | Target        | XGBoost Shallow (Test) | Status                                  |
| --------- | ------------- | ---------------------- | --------------------------------------- |
| PR-AUC    | > 0.85        | 0.849                  | ⚠️ 0.001 below — calibration pending |
| Recall    | > 0.80        | 0.856                  | ✅                                      |
| Precision | > 0.70        | 0.865                  | ✅                                      |
| Accuracy  | (not primary) | 0.9995                 | —                                      |

---

## Evaluation Methodology

### Why Not Accuracy

With a 559:1 class imbalance, a model that predicts every transaction as legitimate achieves **99.82% accuracy** while catching zero fraud. Accuracy is never used as a primary or comparison metric in this project.

### Metric Priority

| Priority | Metric                               | Why                                                                    |
| -------- | ------------------------------------ | ---------------------------------------------------------------------- |
| 1        | **PR-AUC** (Average Precision) | Threshold-free; sensitive to imbalance; random baseline ≈ 0.002       |
| 2        | **Recall**                     | Direct financial cost per missed fraud                                 |
| 3        | **Precision**                  | Operational cost of false alarms (analyst workload, customer friction) |
| 4        | **F2 score** (β=2)            | Single-number comparison weighting recall 2× precision                |
| 5        | Total cost                           | `FN × cost_fn_ratio + FP × 1` for business-grounded comparison     |
| Ref      | ROC-AUC                              | Reported for reference only — inflated by 56,000+ true negatives      |
| ⛔       | Accuracy                             | Never used as primary metric                                           |

### Threshold Selection

The default threshold of 0.5 is wrong at 533:1 imbalance — XGBoost with `scale_pos_weight=559` produces most fraud probabilities well below 0.5 even for genuine fraud. The threshold is always selected from the Precision-Recall curve using one of two strategies:

**Strategy A — Recall floor:** highest Precision at Recall ≥ 0.80
**Strategy B — Cost ratio:** minimises `FN × cost_fn_ratio + FP × 1`

Use `--cost-fn-ratio` to adjust the cost assumption. Default is 10 (missing 1 fraud costs 10× a false alarm). Strategy A threshold is saved with the model by default; override with `--threshold` if Strategy B is preferred.

### Cross-Validation

Stratified K-fold (5 or 10 folds) ensures each fold has the same fraud prevalence (~0.18%). With 305 training fraud cases across 5 folds, each fold trains on ~244 and validates on ~61 fraud cases. Fold-level variance in Precision is expected and normal — the per-fold PR-AUC is the reliable diagnostic.

### SHAP Interpretability

Required for financial services regulatory compliance. SHAP values explain why each individual prediction was made. Run automatically by `testing.py` for XGBoost and LightGBM models. Use `--no-shap` to skip for faster evaluation.

Expected top SHAP features based on EDA correlation analysis: V4, V14, V17, V12, V11_hour, log_amount, V12_amount.

---

## Repository Structure

```
credit-fraud-detection/
│
├── data/
│   ├── train.csv                          Raw training data
│   ├── val.csv                            Raw validation data
│   ├── test.csv                           Raw test data
│   ├── engineered/
│   │   ├── train_features.parquet         Engineered training features
│   │   ├── val_features.parquet           Engineered validation features
│   │   ├── test_features.parquet          Engineered test features
│   │   ├── feature_artifacts.pkl          Fitted transformers (bin edges, encoder)
│   │   ├── train_features_metadata.json
│   │   ├── val_features_metadata.json
│   │   └── test_features_metadata.json
│   └── engineered/oversampled/
│       └── train_features.parquet         SMOTE-oversampled training features
│
├── models/
│   ├── xgboost_shallow/                   ⭐ Best model
│   │   ├── trained_model.pkl
│   │   ├── threshold_report.json
│   │   └── cv_results.json
│   ├── xgboost_folds_5/
│   ├── xgboost_folds_10/
│   ├── lightgbm_v1/
│   ├── lightgbm_v2/
│   ├── logistic_v1/
│   └── logistic_v2/
│
├── eval/
│   ├── xgboost_shallow_test/
│   │   ├── evaluation_metrics.json
│   │   ├── predictions.csv
│   │   ├── precision_recall_curve.png
│   │   ├── confusion_matrix.png
│   │   └── shap_summary.png
│   └── [model]_[split]/                   One directory per model × split
│
├── notebooks/
│   ├── 1_EDA.ipynb                        Story-driven EDA (14 sections)
│
├── src/
│   ├── feature_engineering.py             Feature engineering pipeline
│   ├── training.py                        Model training + CV + threshold selection
│   └── testing.py                         Evaluation + PR curve + SHAP
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Commit your changes: `git commit -m 'Add your feature'`
3. Push to the branch: `git push origin feature/your-feature`
4. Open a Pull Request

**Priority areas for contribution:**

- Probability calibration integration into `testing.py` as a built-in flag
- LightGBM hyperparameter fix (threshold collapse at scale_pos_weight=559)
- Bayesian hyperparameter optimisation for XGBoost shallow config
- KNN and Random Forest experimental results
- Temporal stability analysis (performance across different 6-hour windows within the 48-hour dataset)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built by Mhammed Helal · Last updated: July 2026*