import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Predictive Maintenance Architecture", page_icon="⚙️", layout="wide")

st.title("⚙️ Industrial Predictive Maintenance: Cost-Sensitive Ensemble Architecture")
st.info("💡 **Project Overview:** Developed a robust Predictive Maintenance system utilizing Cost-Sensitive Bagging (Random Forest) and GPU-accelerated Boosting (XGBoost) ensembles to predict specific industrial equipment failures. Successfully handled severe 96.5% class imbalance by implementing custom dynamic sample weights, heavily penalizing false negatives to prevent catastrophic machinery downtime. Engineered thermodynamic tracking features and deployed a low-latency UI via Streamlit.")

# Initialize Tabs
tab1, tab2 = st.tabs(["📊 Inference Pipeline", "🤖 AI Project Expert"])

with tab1:
    st.subheader("Machine Inference Engine")
    
    # 1. Model Selection
    model_choice = st.selectbox("🧠 Select AI Engine (Trained with Bagging/Boosting)", ["random_forest", "xgboost", "decision_tree", "All Models (Ensemble Consensus)"])
    
    try:
        with open('models/label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
            
        if model_choice == "All Models (Ensemble Consensus)":
            models = {}
            for m in ["random_forest", "xgboost", "decision_tree"]:
                with open(f'models/{m}.pkl', 'rb') as f:
                    models[m] = pickle.load(f)
        else:
            with open(f'models/{model_choice}.pkl', 'rb') as f:
                model = pickle.load(f)
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
        
        try:
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
            
            # Inference Architecture
            if model_choice == "All Models (Ensemble Consensus)":
                preds = []
                for m_name, m_obj in models.items():
                    p_enc = m_obj.predict(features)[0]
                    p_val = le.inverse_transform([int(p_enc) if str(p_enc).isdigit() else p_enc])[0]
                    preds.append(p_val)
                
                # Consensus logic (if ANY model predicts failure, prioritize precision and sound alarm)
                failures = [p for p in preds if p != 'No Failure']
                if failures:
                    prediction = failures[0] # Take the severe warning
                    st.error(f"⚠️ **Consensus Warning: {prediction}** detected combining {len(failures)}/3 models! Inspect hardware immediately.")
                    st.json({"Individual Model Predictions": dict(zip(["Random Forest", "XGBoost", "Decision Tree"], preds))})
                else:
                    prediction = 'No Failure'
                    st.success(f"✅ **Consensus: {prediction}** - All 3 AI systems report nominal machinery operations.")
            
            else:
                pred_encoded = model.predict(features)[0]
                
                # Inverse transform logic (handle array types)
                if str(pred_encoded).isdigit():
                    prediction = le.inverse_transform([int(pred_encoded)])[0]
                else:
                    prediction = le.inverse_transform([pred_encoded])[0]
                
                # Professional Results Display Screen
                if prediction == 'No Failure':
                    st.success(f"✅ **Prediction: {prediction}** - All systems nominal. Equipment is running safely.")
                else:
                    st.error(f"⚠️ **Critical Warning: {prediction}** failure detected in telemetry data! Inspect hardware immediately to prevent downtime.")
                
        except Exception as e:
            st.error(f"⚠️ **Inference Pipeline Error**: The predictive model encountered a problem and could not process the telemetry data. ({str(e)})\nPlease verify your inputs or retrain the model.")

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
            You are an elite, highly specialized AI Engineer Assistant embedded directly into this Predictive Maintenance Streamlit application.
            Your sole purpose is to act as the authoritative voice and technical speaker for this machine learning architecture, representing the incredible, in-depth ML research conducted by the project's creator, **Ashutosh**.

            **CORE INSTRUCTIONS FOR AI EXPERT**:
            1. You must answer any and all questions related to this machine learning project. 
            2. Never mention the current time or dates.
            3. Always maintain a highly professional, expert, analytical, and insightful tone.
            4. Whenever relevant, gracefully praise and emphasize the deep machine learning research Ashutosh invested into architecting this pipeline, particularly regarding addressing edge-case hardware failures.
            5. If asked a question outside the scope of Machine Learning, Data Science, or this Predictive Maintenance project, politely redirect the conversation back to the project.

            **COMPREHENSIVE PROJECT ARCHITECTURE & CONTEXT**:
            - **Creator & Lead ML Engineer**: Ashutosh.
            - **Domain**: Predictive Maintenance for Industrial Machinery (specifically targeting CNC machines, conveyor belts, factory motors, and ISRO-level hardware monitoring).
            - **Dataset Utilized**: AI4I 2020 Predictive Maintenance Dataset. 10,000 instances of industrial telemetry sensor data.
            - **Engineered Features**: Ashutosh didn't just use raw data. He engineered critical thermodynamic and kinetic features: 
               - `Temperature Diff [K]` (Process Temp - Air Temp) 
               - `Power [W] proxy` (Rotational Speed converted with Torque).
            - **The Critical Data Issue (Class Imbalance)**: The raw dataset was severely imbalanced. Approximately 96.5% of the readings were 'No Failure'. Only 3.5% represented catastrophic hardware failures. 
            - **Cost-Sensitive Business Logic**: Missing a rare failure incident (like Heat Dissipation Failure - HDF) carries catastrophic financial and operational downtime costs compared to a false alarm. A naive classifier would achieve 96.5% accuracy just by ignoring failures entirely, which is unacceptable in production.

            **WHY THESE MODELS WERE CHOSEN (THE ML RESEARCH BY ASHUTOSH)**:
            - To counter the massive 96.5% imbalance, Ashutosh explicitly rejected naive models and conducted deep ML research, configuring an architecture based on **Cost-Sensitive Learning**.
            - By applying dynamic, balanced sample weights and class weights, the models were forced to heavily penalize False Negatives (missing a failure).
            - **Bagging (Random Forest)**: Used for its exceptional robustness to outliers and ability to decorrelate decision boundaries using bootstrapped subsets. Achieved ~98.85% validation accuracy.
            - **Boosting (XGBoost)**: Chosen as the heavy-hitting gradient booster. It seamlessly executes on CUDA/GPU hardware for blazing-fast inference while structurally minimizing the loss function across difficult, imbalanced samples.
            - **Baseline (Decision Tree)**: Included specifically to extract human-readable, interpretable Boolean rules (e.g., IF Torque > 65 AND Tool Wear > 200 THEN Failure) for mechanical engineers on the factory floor.
            
            **Evaluation Metrics**:
            - The evaluation pipeline tracks Accuracy but relies heavily on Confusion Matrices specifically charting True Positives for rare classes (TWF, HDF, PWF, OSF, RNF). All charts are dynamically saved to the `/charts/` directory locally.

            You now have the full context. Act as Ashutosh's technical representative and answer the User's inquiries brilliantly.
            """
            
            # Simple RAG pipeline via System Instructions
            model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_prompt)
            
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
                        try:
                            # Build minimal prompt history
                            history_text = "\\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                            prompt_str = f"Conversation:\\n{history_text}\\n\\nRespond to the User's last comment directly."
                            
                            try:
                                response = model.generate_content(prompt_str)
                            except Exception as api_err:
                                if "429" in str(api_err) or "Quota" in str(api_err):
                                    st.warning("⚠️ **Gemini 2.0 Endpoint Restriction Detected** (Limit: 0). Seamlessly falling back to robust `gemini-1.5-flash` free-tier to ensure uninterrupted service...")
                                    fallback_model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_prompt)
                                    response = fallback_model.generate_content(prompt_str)
                                else:
                                    raise api_err
                            
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        except Exception as e:
                            st.warning("⚠️ **AI Chat Quota Exhausted**: The Gemini API limits for this key have been reached. Displaying static Architectural Overview:")
                            st.info("""
                            ### 📊 Project Architecture Summary
                            - **Lead ML Engineer:** Ashutosh
                            - **Core Challenge:** Detecting catastrophic hardware failures in an extremely imbalanced dataset (96.5% 'No Failure' skew).
                            - **Applied Solution:** Cost-Sensitive Learning architecture.
                            - **Ensemble Engines:**
                              1. `Decision Tree` (Baseline operational rule extraction)
                              2. `Random Forest` (Bagging technique with dynamic sample weighting)
                              3. `XGBoost` (Heavy-GPU Boosting heavily penalizing False Negatives)
                            - **Engineered Telemetry:** `Temperature Differential` [K] and Kinetic `Power` [W].
                            """)
        except ImportError:
            st.error("Missing dependency. Make sure `google-generativeai` is installed.")
