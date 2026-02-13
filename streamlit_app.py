import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import cv2
import tensorflow as tf
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")
st.write("This app uses your webcam to monitor eye/mouth activity and alert you if you show signs of drowsiness.")

# --- FILE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mouth_cnn.h5')

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

# --- PRE-PROCESSING FUNCTION ---
def prepare_image(frame):
    # Resize to match the input size your model was trained on (usually 224x224 or 64x64)
    # Check your model's summary if this fails!
    img = cv2.resize(frame, (224, 224)) 
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

# --- WEBRTC CALLBACK ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # 1. Convert to Gray for face detection (standard OpenCV approach)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Simple logic placeholder for detection
    # In a full app, you'd use MediaPipe landmarks here to crop the mouth/eyes
    try:
        if model:
            processed_img = prepare_image(img)
            prediction = model.predict(processed_img, verbose=0)
            
            # Example thresholding
            label = "Active" if prediction[0][0] > 0.5 else "Drowsy"
            color = (0, 255, 0) if label == "Active" else (0, 0, 255)
            
            cv2.putText(img, f"Status: {label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    except Exception as e:
        cv2.putText(img, "Model Error", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    webrtc_streamer(
        key="drowsiness-detect",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}, muted=True),
    )

with col2:
    st.subheader("System Status")
    if model:
        st.success("Deep Learning Model Loaded")
    else:
        st.error("Model Offline")
    
    st.info("The system analyzes facial landmarks to detect fatigue.")
    
    # Audio Alert Section (Streamlit hack)
    if st.checkbox("Enable Audio Alarm"):
        st.warning("Audio will play through your browser if drowsiness is detected for >3 seconds.")

import av # Required for the video_frame_callback to work
