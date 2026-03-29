---
title: Industrial Predictive Maintenance
emoji: ⚙️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# ⚙️ Industrial Predictive Maintenance: Cost-Sensitive Ensemble Architecture

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Powered by XGBoost](https://img.shields.io/badge/Powered%20by-XGBoost-1C1C1C?style=for-the-badge&logo=python)](https://xgboost.readthedocs.io/)
[![AI Integration](https://img.shields.io/badge/AI%20Chat-Gemini%202.0%20Flash-4285F4?style=for-the-badge&logo=google)](https://deepmind.google/technologies/gemini/)

💡 **Project Overview**  
Developed a robust Predictive Maintenance system utilizing Cost-Sensitive Bagging (Random Forest) and GPU-accelerated Boosting (XGBoost) ensembles to predict specific industrial equipment failures. Successfully handled severe 96.5% class imbalance by implementing custom dynamic sample weights, heavily penalizing false negatives to prevent catastrophic machinery downtime. Engineered thermodynamic tracking features and deployed a low-latency UI via Streamlit.

**Lead ML Engineer:** Ashutosh

---

## 🏗️ Technical Architecture

### 1. The Imbalance Problem & Cost-Sensitive Learning
Industrial machinery data (AI4I 2020) is massively skewed: **96.5%** of telemetry reads show 'No Failure'. A naive model would achieve near-perfect accuracy by simply never predicting a breakdown, which is catastrophic in a real-world manufacturing plant.
- **Solution:** This architecture enforces extreme class weights and balanced sample penalizations on Tree-based ensembles, overriding the imbalance and forcing the ML to memorize rare edge-cases (e.g., Heat Dissipation Failures).

### 2. The Machine Learning Engine
- **Random Forest (Bagging):** Utilizes heavily bootstrapped subsets to decorrelate decision boundaries, maximizing generalized recall across varying failure types without overfitting.
- **XGBoost (Boosting):** The core gradient booster. Seamlessly processes the complex non-linear relationships of the newly engineered thermodynamic features.
- **Decision Tree (Baseline):** Maintained specifically to extract human-readable, interpretable Boolean rules for mechanical engineers on the factory floor.
- **All Models (Ensemble Consensus):** A proprietary UI feature evaluating real-time telemetry against all 3 models instantly, triggering a Consensus Alarm if *any* underlying architecture detects impending structural failure.

### 3. Feature Engineering
Raw data is rarely enough. The feature space was structurally augmented in the processing pipeline:
- `Temperature Diff [K]` = Process Temperature - Air Temperature
- `Power Proxy [W]` = Rotational Speed (RPM) × Torque (Nm)

### 4. 🤖 RAG Project Assistant (Gemini 2.0)
This repository ships with an integrated Generative AI expert. The UI includes a secondary tab that communicates directly with `gemini-2.0-flash`. The model has been precisely prompted with Ashutosh's technical architecture, meaning users can fluidly interrogate the dashboard on *why* certain algorithms were chosen, or what specific features were engineered. 
*(Includes dynamic API exhaustion fallbacks and robust rate-limit catching natively in Python)*.

---

## 🚀 Quick Start (Local Setup)

If cloning this repository from GitHub, you will need Git LFS installed to pull the heavy `.pkl` ensemble models.

```bash
# Clone the repository
git clone https://github.com/yourusername/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance

# Create and activate environment
conda create -n ml-env python=3.9 -y
conda activate ml-env

# Install dependencies
pip install -r requirements.txt

# Securely configure the Gemini AI Expert (Create a .env file)
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Launch the Application
streamlit run app.py
```

## ☁️ Hugging Face Deployment
This repository is natively architected for **Hugging Face Spaces**. The YAML frontmatter at the top of this `README.md` automatically configures the Space environment upon push.
1. Connect your Github to a new HF Streamlit Space.
2. In HF Settings -> Variables and Secrets, add `GEMINI_API_KEY`.
3. The Space will automatically deploy the robust application!
