<div align="center">
  
# ⚙️ Predictive Maintenance for Industrial Machines

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU--Enabled-green.svg)](https://xgboost.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)](https://scikit-learn.org/)

*An end-to-end Machine Learning solution predicting equipment failure types using sensor telemetry.*

</div>

## 📌 Overview

Unplanned equipment failure causes massive downtime. Traditional "fix when broken" maintenance is reactive and expensive. This project utilizes the **AI4I 2020 Predictive Maintenance Dataset** to predict specific failure types (Tool Wear, Heat Dissipation, Power Failure, Overstrain) *before* they occur.

This repository is built with simplicity, clean code, and production deployment in mind, featuring:
- **Bagging & Boosting Ensembles**: Using Random Forest and XGBoost.
- **GPU Acceleration**: XGBoost automatically hooks into CUDA hardware for blazing-fast training, gracefully falling back to CPU.
- **Feature Engineering**: Deriving thermodynamic and kinetic features (e.g. proxy power and temperature diffs).
- **Interactive UI**: An intuitive Streamlit app for real-time inference.

---

## 📂 Repository Structure

```text
📦 02_predictive_maintenance
 ┣ 📂 data
 ┃ ┣ 📂 raw                  # Original dataset (ai4i2020.csv)
 ┃ ┗ 📂 processed            # Generated clean dataset
 ┣ 📂 models                 # Trained serialized models & dependencies
 ┣ 📂 charts                 # Generated evaluation metrics & visual charts
 ┣ 📜 01_eda_and_preprocessing.ipynb # Data cleaning & Feature engineering
 ┣ 📜 02_model_training.ipynb        # Ensemble model training
 ┣ 📜 03_evaluation.ipynb            # Performance charting & validation
 ┣ 📜 app.py                         # Streamlit Interactive Dashboard
 ┣ 📜 PROJECT_PROBLEM.md             # Detailed domain context
 ┣ 📜 requirements.txt               # Pipeline dependencies
 ┗ 📜 README.md                      # Professional Project Guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebooks (Sequentially)
Run `.ipynb` files to generate datasets, models, and evaluation charts down the pipeline:
1. `01_eda_and_preprocessing.ipynb` 
2. `02_model_training.ipynb` 
3. `03_evaluation.ipynb`

### 3. Launch the Web Interface
Fire up the web application to make real-time hardware health predictions.
```bash
streamlit run app.py
```

## 🧠 ML Architecture
- **Preprocessing**: Handling multi-target flags, extracting `Temperature Diff` and `Proxy Power` features.
- **Ensemble Strategy**: We compare decision boundaries of shallow trees, bagging (Random Forest), and gradient boosting (XGBoost).
- **Inference App**: The Streamlit interface loads the desired standard Python `pickle` model and label encoder, dynamically preprocessing real-time telemetry from user inputs and displaying early-warning fault detection results.

## 📊 Evaluation
Check out the automatically generated `/charts` directory upon successful model training to view the Confusion Matrices and multi-model performance benchmarks. We prioritize capturing high-risk faults over general accuracy to minimize catastrophic equipment breakdown.
