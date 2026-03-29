import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="centered")

st.title("⚙️ Predictive Maintenance Multi-Model App")
st.markdown("*Predict machine failure types gracefully before they occur using Cost-Sensitive Machine Learning Ensembles.*")

# 1. Model Selection
model_choice = st.selectbox("🧠 Select AI Engine (Trained with Bagging/Boosting)", ["random_forest", "xgboost", "decision_tree"])

try:
    with open(f'models/{model_choice}.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Pre-trained models missing. Please run the Jupyter pipelines (`01` and `02`) first!")
    st.stop()

st.divider()
st.subheader("🔧 Telemetry Input")

# 2. Real-Time Inputs
col1, col2 = st.columns(2)
with col1:
    air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
    process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
    rpm = st.number_input("Rotational speed [rpm]", value=1500.0, step=10.0)
with col2:
    torque = st.number_input("Torque [Nm]", value=40.0, step=1.0)
    tool_wear = st.number_input("Tool wear [min]", value=0, step=1)

# 3. Predict & Engineer Features on the fly
st.divider()
if st.button("🚀 Predict Hardware Status", use_container_width=True):
    
    # Feature Engineering mirroring Notebook 01
    temp_diff = process_temp - air_temp
    power_w = rpm * torque * 0.10472
    
    features = pd.DataFrame([{
        'Air temperature': air_temp,
        'Process temperature': process_temp,
        'Temperature Diff': temp_diff,
        'Rotational speed': rpm,
        'Torque': torque,
        'Power': power_w,
        'Tool wear': tool_wear
    }])
    
    # Inference
    pred_encoded = model.predict(features)[0]
    
    # Inverse transform logic (handle array types)
    try:
        if str(pred_encoded).isdigit():
            prediction = le.inverse_transform([int(pred_encoded)])[0]
        else:
            prediction = le.inverse_transform([pred_encoded])[0]
    except Exception:
        prediction = le.inverse_transform([pred_encoded])[0]
    
    # Professional Results Display Screen
    if prediction == 'No Failure':
        st.success(f"✅ **Prediction: {prediction}** - All systems nominal. Equipment is running safely.")
    else:
        st.error(f"⚠️ **Critical Warning: {prediction}** failure detected in telemetry data! Inspect hardware immediately to prevent downtime.")
