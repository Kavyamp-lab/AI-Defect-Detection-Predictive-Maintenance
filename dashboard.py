import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import time
import sys
import os

# Ensure src modules can be loaded
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import DefectDetectorCNN, PredictiveMaintenanceLSTM
from src.data_preprocessing import get_image_transforms

st.set_page_config(page_title="Smart Manufacturing Edge Dashboard", layout="wide")

# --- MODEL LOADING (Cached for performance) ---
@st.cache_resource
def load_models():
    # Load CNN
    cnn = DefectDetectorCNN(pretrained=False)
    # Note: In production, uncomment the line below to load trained weights
    # cnn.load_state_dict(torch.load('cnn_defect_model.pth', map_location='cpu'))
    cnn.eval()
    
    # Load LSTM (Assuming 4 sensor features: Temp, Vibration, Pressure, Speed)
    lstm = PredictiveMaintenanceLSTM(input_size=24)
    lstm.load_state_dict(torch.load("lstm_rul_model.pth", map_location="cpu"))
    # lstm.load_state_dict(torch.load('lstm_rul_model.pth', map_location='cpu'))
    lstm.eval()
    
    return cnn, lstm

cnn_model, lstm_model = load_models()
transform = get_image_transforms(is_train=False)

# --- DASHBOARD UI ---
st.title("🏭 Smart Manufacturing: AI Control Center")
st.markdown("Real-time Defect Detection & Predictive Maintenance Monitoring")

col1, col2 = st.columns(2)

# --- 1. REAL-TIME DEFECT DETECTION ---
with col1:
    st.header("🔍 Visual Defect Detection")
    st.write("Upload or capture production line images to detect anomalies.")
    
    uploaded_file = st.file_uploader("Upload Component Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Camera Feed", use_column_width=True)
        
        # Inference
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            prob = cnn_model(input_tensor).item()
            
        st.subheader("Analysis Results:")
        if prob > 0.6:
            st.error(f"🚨 DEFECT DETECTED (Confidence: {prob*100:.2f}%)")
            st.toast("Alert: Defective product ejected from line.", icon="🚨")
        else:
            st.success(f"✅ COMPONENT NORMAL (Defect Prob: {prob*100:.2f}%)")

# --- 2. PREDICTIVE MAINTENANCE (SENSOR TIME-SERIES) ---
with col2:
    st.header("⚙️ Machine Health (Predictive Maintenance)")
    st.write("Live sensor telemetry and Remaining Useful Life (RUL) prediction.")
    
    # Simulate Live Sensor Data (Temperature, Vibration, Pressure, Speed)
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = pd.DataFrame(columns=['Time', 'Temp', 'Vibration', 'Pressure', 'Speed'])
    
    # Generate new synthetic data point
    new_time = pd.Timestamp.now()
    new_data = {
        'Time': new_time,
        'Temp': np.random.normal(70, 5),
        'Vibration': np.random.normal(2.0, 0.5),
        'Pressure': np.random.normal(101, 2),
        'Speed': np.random.normal(1500, 50)
    }
    
    df = st.session_state.sensor_data
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    if len(df) > 50: df = df.iloc[-50:] # Keep last 50 points
    st.session_state.sensor_data = df
    
    # Plotting Sensors
    fig = px.line(df, x='Time', y=['Temp', 'Vibration'], title="Live Telemetry (Temp & Vib)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Inference (Require sequence length of e.g. 10)
    window_size = 10
    if len(df) >= window_size:
        recent_seq = df[['Temp', 'Vibration', 'Pressure', 'Speed']].iloc[-window_size:].values
        seq_tensor = torch.tensor(recent_seq, dtype=torch.float32).unsqueeze(0) # Batch size 1
        
        with torch.no_grad():
            rul_prediction = lstm_model(seq_tensor).item()
            
        # Display RUL Alert
        st.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{abs(rul_prediction):.1f} Hrs")
        
        # Add a mock condition to show how alerts work based on model
        if abs(rul_prediction) < 10.0: # If less than 10 hours left
            st.warning("⚠️ Warning: Machine requires maintenance soon.")

# Refresh button to simulate streaming
if st.button("Fetch Live Sensor Data"):
    st.rerun()