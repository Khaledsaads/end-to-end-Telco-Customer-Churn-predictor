# Telco Customer Churn Prediction

A binary classification project predicting which telecom customers are likely to churn, using the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features).

The project focuses on **EDA-driven feature engineering for an imbalanced classification problem**, comparing five models through a shared scikit-learn pipeline.

## Final Results

Evaluated on a held-out test set (10% of data), ranked by F1-score:

| Model | AP | Accuracy | F1-score | Precision | Recall |
|---|---|---|---|---|---|
| **XGBoost** | 0.623 | 0.714 | **0.613** | 0.464 | **0.903** |
| Logistic Regression | 0.624 | 0.744 | 0.607 | 0.493 | 0.790 |
| LightGBM | 0.624 | 0.792 | 0.592 | 0.582 | 0.602 |
| SGDClassifier | 0.580 | 0.734 | 0.587 | 0.480 | 0.756 |
| Random Forest | **0.630** | **0.798** | 0.527 | **0.637** | 0.449 |

- **XGBoost** gives the best F1-score and recall — best at catching customers who will actually churn (important for retention campaigns), at the cost of more false positives.
- **Random Forest** gives the best accuracy, average precision, and precision — fewer false alarms, but misses more true churners.
- Class imbalance (~73.5% retained vs. 26.5% churned) was handled via `class_weight='balanced'` (linear models, Random Forest) and `scale_pos_weight` (XGBoost), tuned through `RandomizedSearchCV`/`GridSearchCV` with `average_precision` as the scoring metric.

## Workflow

### 1. Exploratory Data Analysis
- **Target**: Churn is imbalanced (73.5% No / 26.5% Yes) — confirmed the need for class-weighting strategies before training.
- **Data quality**: `TotalCharges` was stored as an object with 11 blank values; converting to numeric revealed these belong to brand-new customers (`tenure = 0`), so they were treated as valid zeros rather than errors.
- **Separation analysis**: `tenure` is the strongest separator between churned and retained customers (median 10 vs. 38), `TotalCharges` is moderately separating, and `MonthlyCharges` is the weakest.
- **Correlation analysis**: Internet-related service features (`OnlineSecurity`, `TechSupport`, etc.) correlate meaningfully with churn, while phone-related features (`PhoneService`, `MultipleLines`) show weak correlation and were candidates for removal.

### 2. Cleaning & Preprocessing
- Dropped `customerID` (unique identifier, no predictive value).
- Removed duplicate rows.
- Imputed the 11 missing `TotalCharges` values with `0` (new customers).
- Applied a 3-way split: 80% train / 10% validation / 10% test.
- Log-transformed `tenure` (`log(tenure + 1)`) to reduce skew.

### 3. Encoding & Feature Engineering
Implemented as a `Pipeline` of `ColumnTransformer` + a custom `FE` transformer + `StandardScaler`:

| Step | Technique | Applied to |
|---|---|---|
| Nominal encoding | `OneHotEncoder` (drop first) | `gender`, `Dependents`, `Partner`, `PaperlessBilling`, `PaymentMethod`, `PhoneService` |
| Ordinal encoding | `OrdinalEncoder` with explicit category order | `InternetService`, `Contract`, and 6 internet add-on features (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) |
| Custom FE | `FE` transformer | Consolidates `"No internet service"` into `"No"` for add-on features (since `InternetService` already captures that signal); engineers new features; drops redundant raw columns |
| Scaling | `StandardScaler` | All numeric features |

Engineered features:
- `average_charges` = `TotalCharges / (tenure + 1)`
- `monthly_to_total` = `MonthlyCharges / (TotalCharges + 1)`
- `Family` = `Partner_Yes + Dependents_Yes`

The fitted pipeline is serialized with `pickle` (`Models/pipeline.pickle`) for reuse on new data.

### 4. Modeling
Five classifiers were trained and evaluated with **Average Precision (AP)** as the primary scoring metric (suited to imbalanced classification):

- **Linear baselines**: `SGDClassifier` (log loss, `class_weight='balanced'`), `LogisticRegression` (`class_weight='balanced'`)
- **Ensembles**: `RandomForestClassifier` tuned via `RandomizedSearchCV` then `GridSearchCV`; `XGBClassifier` tuned via `RandomizedSearchCV` with `scale_pos_weight`; `LGBMClassifier` tuned via `RandomizedSearchCV`
- The best XGBoost model is serialized with `pickle` (`Models/xgbc.pickle`)

## Tech Stack

- **Data manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Preprocessing**: `scikit-learn` (`Pipeline`, `ColumnTransformer`, custom transformers, `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`)
- **Modeling**: `scikit-learn` (`SGDClassifier`, `LogisticRegression`, `RandomForestClassifier`), `xgboost`, `lightgbm`
- **Tuning**: `RandomizedSearchCV`, `GridSearchCV`
- **Persistence**: `pickle`

## Project Structure

```
.
├── Teko_Customer_Churn.ipynb              # Main notebook (EDA → preprocessing → modeling)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset (from Kaggle)
├── Models/
│   ├── pipeline.pickle                     # Fitted preprocessing pipeline
│   └── xgbc.pickle                         # Best XGBoost model
└── README.md
```

## How to Run

1. Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
   ```
3. Run the notebook top to bottom. The fitted pipeline and best model are saved to `Models/` for reuse.

## Notes & Future Work

- Recall vs. precision trade-off matters here: if the business cost of missing a churner is high, XGBoost's higher recall (0.90) is preferable; if false retention-campaign targeting is costly, Random Forest's higher precision (0.64) is preferable.
- Potential next steps: build a simple cost-based threshold tuning step (rather than the default 0.5 cutoff) to directly optimize for business cost of false negatives vs. false positives; add SHAP-based feature importance analysis to explain individual churn predictions; wrap the saved pipeline + model in a small FastAPI service for inference.
