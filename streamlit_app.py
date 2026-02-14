import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import os
import av

# --- CONFIGURATION ---
st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("🚗 Real-Time Drowsiness Detection")

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'mouth_cnn.h5')
LANDMARKER_PATH = os.path.join(BASE_DIR, 'face_landmarker.task')

# Load TensorFlow Model
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = load_keras_model()

# --- WEB-RTC CALLBACK ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    
    # 1. MediaPipe Face Detection
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    
    status = "Searching for Face..."
    color = (255, 255, 255)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 2. Extract Mouth Region (landmarks 0, 13, 14, 17 are around the lips)
            # We'll use a bounding box approach for simplicity
            mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
            x_coords = [int(face_landmarks.landmark[i].x * w) for i in mouth_indices]
            y_coords = [int(face_landmarks.landmark[i].y * h) for i in mouth_indices]
            
            x1, y1 = min(x_coords) - 10, min(y_coords) - 10
            x2, y2 = max(x_coords) + 10, max(y_coords) + 10
            
            # Crop and Predict
            try:
                mouth_crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if mouth_crop.size > 0:
                    # Prepare for CNN (Assumes 224x224 input)
                    input_img = cv2.resize(mouth_crop, (224, 224))
                    input_img = input_img / 255.0
                    input_img = np.expand_dims(input_img, axis=0)
                    
                    prediction = model.predict(input_img, verbose=0)
                    
                    # 3. Logic based on your model's output
                    # Assuming 0 = Drowsy/Yawn, 1 = Active
                    if prediction[0][0] < 0.5:
                        status = "DROWSY ALERT!"
                        color = (0, 0, 255) # Red
                    else:
                        status = "Active"
                        color = (0, 255, 0) # Green
                
                # Draw box around mouth
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            except Exception as e:
                pass

    # Overlay Text
    cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI START ---
webrtc_streamer(
    key="drowsiness",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.sidebar.info("This system uses MediaPipe for face tracking and a custom CNN for mouth analysis.")
