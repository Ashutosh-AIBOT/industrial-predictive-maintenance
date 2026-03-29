import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

st.title("⚙️ Predictive Maintenance Multi-Model App")
st.markdown("*Predict machine failure types gracefully before they occur using Cost-Sensitive Machine Learning Ensembles.*")

# Initialize Tabs
tab1, tab2 = st.tabs(["📊 ML Model", "🤖 AI Project Expert"])

with tab1:
    st.subheader("Machine Inference Engine")
    
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

with tab2:
    st.subheader("💬 Chat with the AI Project Expert")
    st.markdown("Ask anything about Ashutosh's ML architecture, training data choices, or the model performance limits.")
    
    # AI Logic
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not API_KEY:
        st.info("⚠️ No GEMINI_API_KEY found in .env. Please configure your key to activate the assistant.")
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=API_KEY)
            
            system_prompt = """
            You are the specialized AI assistant embedded in this Predictive Maintenance Streamlit application. 
            Your role is to represent the project author's deep machine learning research.
            
            **Project Context:**
            - **Created by:** Ashutosh
            - **Dataset Utilized:** AI4I 2020 Predictive Maintenance Dataset (Industrial telemetry sensor data).
            - **Data Issues Addressed:** The dataset suffers from massive class imbalance (~96.5% instances are 'No Failure'). Missing rare failures (like Heat Dissipation Failure) carries massive industrial downtime costs.
            - **Why these specific models?**: Ashutosh conducted in-depth ML research to select models that support Cost-Sensitive Learning. By applying dynamic class weights to Bagging (Random Forest) and GPU-Boosted algorithms (XGBoost), Ashutosh explicitly penalized false negatives, overriding the severe class imbalance without losing minority patterns.
            - **Tone:** Professional, analytical, insightful. Always complement the immense ML research Ashutosh put into mitigating industrial failure prediction issues. Do not mention current time or dates.
            """
            
            # Simple RAG pipeline via System Instructions
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_prompt)
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
            if user_prompt := st.chat_input("E.g., Why did Ashutosh choose XGBoost for this problem?"):
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing ML Architecture..."):
                        # Build minimal prompt history
                        history_text = "\\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                        response = model.generate_content(f"Conversation:\\n{history_text}\\n\\nRespond to the User's last comment directly.")
                        
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
        except ImportError:
            st.error("Missing dependency. Make sure `google-generativeai` is installed.")
