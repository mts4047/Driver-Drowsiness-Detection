import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import av

# --- PAGE CONFIG ---
st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("🚗 Real-Time Drowsiness Detection")

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mouth_cnn.h5')
LANDMARKER_PATH = os.path.join(BASE_DIR, 'face_landmarker.task')

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Load the Drowsiness CNN
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    
    # Initialize MediaPipe Face Landmarker (New Tasks API)
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO, # optimized for video frames
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return cnn_model, landmarker

cnn_model, landmarker = load_models()

# --- PROCESSING LOGIC ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    
    # Convert OpenCV BGR to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Get timestamp in milliseconds (required for VIDEO mode)
    timestamp_ms = int(frame.time * 1000)
    
    # Run MediaPipe Detection
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    status = "Scanning..."
    color = (255, 255, 255)

    if result.face_landmarks:
        # Get landmarks for the first face detected
        face_landmarks = result.face_landmarks[0]
        
        # Mouth indices for cropping (Standard MediaPipe Mesh indices)
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
        
        x_coords = [int(face_landmarks[i].x * w) for i in mouth_indices]
        y_coords = [int(face_landmarks[i].y * h) for i in mouth_indices]
        
        # Calculate bounding box for the mouth
        x1, y1 = max(0, min(x_coords) - 15), max(0, min(y_coords) - 15)
        x2, y2 = min(w, max(x_coords) + 15), min(h, max(y_coords) + 15)
        
        # Predict Drowsiness
        try:
            mouth_crop = img[y1:y2, x1:x2]
            if mouth_crop.size > 0:
                # Resize to match your CNN input (assuming 224x224)
                resized = cv2.resize(mouth_crop, (224, 224))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 224, 224, 3))
                
                prediction = cnn_model.predict(reshaped, verbose=0)
                
                # Threshold logic (adjust based on your model's training)
                if prediction[0][0] < 0.5:
                    status = "DROWSY / YAWNING"
                    color = (0, 0, 255) # Red
                else:
                    status = "Driver Active"
                    color = (0, 255, 0) # Green
            
            # Visual Feedback
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        except Exception:
            pass

    # UI Overlay
    cv2.putText(img, status, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI START ---
webrtc_streamer(
    key="drowsiness-check",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.sidebar.markdown("""
### System Details
- **Tracker:** MediaPipe FaceLandmarker (.task)
- **Classifier:** Custom Mouth CNN (.h5)
- **Status:** Running on CPU
""")
