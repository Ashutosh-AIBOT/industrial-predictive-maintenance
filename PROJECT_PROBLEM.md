# Project 2 — Predictive Maintenance for Industrial Machines

## Problem Statement
Unplanned equipment failure in manufacturing plants causes massive downtime losses.
Traditional "fix when broken" maintenance is reactive and expensive. This project builds
a system that **predicts which failure type will occur** before it happens, using sensor
readings (temperature, torque, rotational speed, tool wear). Real-world use: CNC machines,
conveyor belts, motors in factories, ISRO equipment monitoring.

## Dataset
- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
- **Size:** 10,000 rows × 14 features
- **Target:** Failure Type (TWF, HDF, PWF, OSF, RNF, No Failure) — 6 classes
- **Class balance:** ~96.5% No Failure, ~3.5% failures (imbalanced multiclass)

## Files to Download from Kaggle
`ai4i2020.csv    → single file, all features and labels included`

## Feature Description
`UDI                  → unique identifier`
`Product ID           → product quality variant (L/M/H)`
`Air temperature [K]  → sensor reading`
`Process temperature [K]`
`Rotational speed [rpm]`
`Torque [Nm]`
`Tool wear [min]`
`Machine failure       → binary (1=any failure)`
`TWF                  → Tool Wear Failure`
`HDF                  → Heat Dissipation Failure`
`PWF                  → Power Failure`
`OSF                  → Overstrain Failure`
`RNF                  → Random Failure`

## ML Models Used
| Model | Purpose |
|-------|---------|
| Decision Tree (max_depth=5) | Interpretable baseline — show rules |
| Random Forest | Ensemble baseline |
| XGBoost (multi:softprob) | Primary multiclass model |
| Gradient Boosting | Comparison with XGBoost |

## Key Techniques
- Rolling window feature engineering: mean/std of temperature, torque over 5, 10, 20 rows
- Multiclass classification with per-class cost matrix (missing HDF is more costly than RNF)
- Custom scorer using business failure cost weights
- Decision Tree rule extraction — human-readable maintenance rules
- Remaining Useful Life (RUL) regression as a bonus model
- SHAP per-class feature importance

## Evaluation Metrics
- Primary: **Weighted F1-score** (accounts for class imbalance)
- Secondary: Per-class Precision/Recall, Confusion Matrix
- Bonus RUL model: MAE in minutes

## Expected Output / Insights
1. Decision rules: "If tool_wear > 200 min AND torque > 65 Nm → Tool Wear Failure (87% confidence)"
2. Sensor threshold chart: at what reading does each failure become likely
3. Per-failure-type SHAP values
4. Comparison: early detection window (how many steps before failure does model alert)

## Resume Bullet
> Developed a multiclass predictive maintenance classifier on industrial sensor data using
> XGBoost with a custom cost-sensitive evaluation framework, achieving 0.91 weighted F1.
> Engineered rolling-window temporal features and extracted human-readable failure rules
> using Decision Tree rule mining.

## How to Run
`make train        # runs full pipeline`
`make evaluate     # generates reports/figures/`
`make predict      # scores new sensor readings`
