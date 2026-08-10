# NHS A&E 4-Hour Breach Prediction

A machine learning pipeline that predicts whether an NHS Trust will breach the **4-hour A&E waiting time target** in a given month, using historical NHS Accident & Emergency (A&E) attendance and admissions data.

## Overview

The NHS operational standard requires that at least 95% of patients attending A&E are admitted, transferred, or discharged within 4 hours. This project builds a binary classifier that flags months where a Trust's **breach rate exceeds 5%**, using attendance volumes, admissions, and engineered time-series features.

The notebook covers the full pipeline: data cleaning, feature engineering, class balancing, model comparison, hyperparameter-free baselines, SHAP explainability, and subgroup (fairness) analysis.

## Dataset

- **Source**: NHS England A&E Attendances and Emergency Admissions statistics (merged monthly Trust-level dataset)
- **Granularity**: One row per NHS Trust per month
- **Key fields**:
  - A&E attendances (Type 1 / Type 2 / Other)
  - Attendances breaching the 4-hour target
  - Emergency admissions via A&E
  - Patients waiting 4–12 hrs / 12+ hrs from decision-to-admit
- **Target variable**: `breach` — 1 if a Trust's monthly breach rate (over-4hr attendances ÷ total attendances) exceeds 5%, else 0

> The raw CSV is expected at `NHS and AE Dataset/merged_ae_dataset.csv` (the notebook was originally built for Google Colab + Google Drive).

## Pipeline

1. **Data loading & cleaning** — drop duplicate columns, parse NHS period strings (e.g. `MSitAE-JANUARY-2021`) into dates, coerce numeric fields
2. **Target engineering** — compute `breach_rate` and binary `breach` label
3. **Feature engineering**:
   - Time features: month, quarter, year, `is_winter`, `is_summer`
   - Rolling 3-month averages of attendances and breaches
   - Lag features (previous 1–2 months' breach rate per Trust)
   - Derived ratios (Type 1 attendance proportion, admission pressure, long-wait ratio)
   - Encoded Trust ID
4. **Exploratory data analysis** — national breach rate over time, seasonality, and other diagnostic plots
5. **Class balancing** — `StandardScaler` + `SMOTE` oversampling to address class imbalance
6. **Model comparison** (10-fold stratified cross-validation, scored on F1, ROC-AUC, Recall, Precision):
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - K-Nearest Neighbours
7. **Final evaluation** — hold-out test set metrics, confusion matrix, ROC curve for the best-performing model(s) (Random Forest and XGBoost)
8. **Explainability** — SHAP global importance and beeswarm plots to interpret model predictions
9. **Fairness / subgroup analysis** — model performance broken down by season (winter vs. non-winter)
10. **Artifacts saved** — trained model (`best_model_rf.pkl`), scaler (`scaler.pkl`), and results table (`model_results.csv`)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap joblib
```

## Usage

1. Place `merged_ae_dataset.csv` in a folder named `NHS and AE Dataset/`.
2. If running outside Google Colab, replace the `google.colab.drive` mount step and update the file paths (`ae_path`, `SAVE`) to point to your local directory.
3. Run the notebook `NHSproject_sandy.ipynb` cell by cell.
4. Outputs (plots, trained model, scaler, results CSV) are saved to the same data folder.

## Outputs

| File | Description |
|---|---|
| `model_comparison.png` | Bar chart comparing F1, ROC-AUC, Recall, and Precision across models |
| `shap_importance.png` | Global SHAP feature importance |
| `shap_beeswarm.png` | SHAP beeswarm plot showing direction/magnitude of feature impact |
| `best_model_rf.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted `StandardScaler` used for preprocessing |
| `model_results.csv` | Cross-validated performance metrics for all models |

## Notes

- The notebook was developed in Google Colab; file paths assume a mounted Google Drive and will need adjusting for local or other cloud environments.
- Class imbalance is handled via SMOTE — evaluate results on the untouched hold-out/raw split (as done in the fairness analysis) to get a realistic picture of real-world performance.

## License

Add a license of your choice (e.g. MIT) here.
