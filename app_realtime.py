"""
Real-Time Handwritten Digit Detection - Fixed for Streamlit Cloud
==================================================================
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import time
import av
import warnings
import asyncio
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

# Suppress noisy asyncio warnings from aioice
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Real-Time Digit Detection", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2rem; text-align: center; color: #1E88E5; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (Global reference for VideoProcessor)
# ─────────────────────────────────────────────────────────────────────────────
MODEL = None

@st.cache_resource
def load_model():
    paths = [
        "best_cnn_model/best_cnn_model(1).keras",
        "best_cnn_model/best_cnn_model.keras",
        "cnn_enhanced_model.keras",
    ]
    for path in paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path
    return None, None

model, model_path = load_model()
if model is not None:
    MODEL = model
    st.success(f"✅ Model loaded: `{os.path.basename(model_path)}`")
else:
    st.error("❌ Model not found! Please upload .keras file.")
    uploaded = st.file_uploader("Upload Model", type=["keras"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(uploaded.read())
            model = tf.keras.models.load_model(tmp.name)
            MODEL = model
            st.success("Model loaded!")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class DigitDetector(VideoProcessorBase):
    def __init__(self):
        self.last_pred = None
        self.last_conf = 0.0
        self.last_time = 0
        self.status = "Waiting..."

    def recv(self, frame):
        if MODEL is None:
            return frame
            
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # Predict every 0.5 seconds
        if current_time - self.last_time > 0.5:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) > 127:
                    gray = cv2.bitwise_not(gray)
                
                small = cv2.resize(gray, (224, 224))
                _, thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    if w > 20 and h > 20:
                        pad = 20
                        y1, y2 = max(0, y-pad), min(224, y+h+pad)
                        x1, x2 = max(0, x-pad), min(224, x+w+pad)
                        crop = small[y1:y2, x1:x2]
                        
                        digit_img = cv2.resize(crop, (28, 28))
                        tensor = (digit_img / 255.0).reshape(1, 28, 28, 1).astype(np.float32)
                        
                        preds = MODEL.predict(tensor, verbose=0)[0]
                        self.last_pred = int(np.argmax(preds))
                        self.last_conf = float(np.max(preds)) * 100
                        self.status = f"Digit: {self.last_pred}"
                        
                        scale_x = img.shape[1] / 224
                        scale_y = img.shape[0] / 224
                        cv2.rectangle(img, (int(x1*scale_x), int(y1*scale_y)), 
                                      (int(x2*scale_x), int(y2*scale_y)), (0, 255, 0), 3)
                    else:
                        self.status = "Too Small"
                        scale_x = img.shape[1] / 224
                        scale_y = img.shape[0] / 224
                        cv2.rectangle(img, (int(x*scale_x), int(y*scale_y)), 
                                      (int((x+w)*scale_x), int((y+h)*scale_y)), (0, 0, 255), 2)
                else:
                    self.status = "No Digit Found"
                    
            except Exception as e:
                self.status = f"Error: {str(e)[:10]}"
            
            self.last_time = current_time

        cv2.rectangle(img, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(img, self.status, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header"> Real-Time Digit Detection</h1>', unsafe_allow_html=True)

if MODEL is not None:
    st.markdown("### 📹 Live Camera")
    st.caption("Look for the **Green Box** (Digit Found) on the video.")
    
    # Robust RTC Configuration for Streamlit Cloud
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            # Public TURN server (free, but may be slow)
            {
                "urls": "turn:openrelay.metered.ca:80",
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }

    ctx = webrtc_streamer(
        key="digit-cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DigitDetector,
        media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
        rtc_configuration=rtc_config,
    )

    if ctx.state.playing:
        st.success("🟢 Camera Active")
    else:
        st.warning("⚪ Camera Stopped - Check permissions or network")

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD FALLBACK
# ────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📤 Or Upload an Image")
uploaded = st.file_uploader("Upload", type=["png", "jpg"])
if uploaded and MODEL:
    img = Image.open(uploaded).convert("L")
    arr = np.array(img.resize((28, 28))) / 255.0
    pred = MODEL.predict(arr.reshape(1, 28, 28, 1), verbose=0)[0]
    
    col1, col2 = st.columns(2)
    with col1: st.image(img, width=150)
    with col2:
        digit = int(np.argmax(pred))
        conf = float(np.max(pred)) * 100
        st.markdown(f"### Prediction: **{digit}** ({conf:.1f}%)")
