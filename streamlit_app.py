import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import tensorflow as tf
import numpy as np

# Load your model (ensure the path matches your repo)
model = tf.keras.models.load_model('models/cnnCat2.h5')

st.title("Driver Drowsiness Detection")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # --- INSERT VILYA'S DETECTION LOGIC HERE ---
    # 1. Convert to grayscale
    # 2. Detect faces/eyes
    # 3. Predict with model
    # 4. Draw rectangles/text on 'img'
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="drowsiness", video_frame_callback=video_frame_callback)
