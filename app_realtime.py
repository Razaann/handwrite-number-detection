"""
Real-Time Handwritten Digit Detection - Streamlit Web App
==========================================================
Uses phone camera for real-time digit recognition
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import tempfile
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Digit Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS FOR MOBILE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        text-align: center;
        color: #1E88E5;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .prediction-digit {
        font-size: 5rem;
        font-weight: bold;
        margin: 0;
    }
    .prediction-conf {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-stopped {
        color: #FF5252;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load CNN model"""
    paths = [
        "best_cnn_model/best_cnn_model(1).keras",
        "best_cnn_model/best_cnn_model.keras",
        "cnn_enhanced_model.keras",
        "cnn_digit_model.keras",
        "/content/drive/MyDrive/Dataset Colab/NumberHandWritten/cnn_enhanced_model.keras",
    ]
    
    for path in paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path), path
    
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    """
    Preprocess camera frame to MNIST format
    frame: numpy array from camera (BGR format)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Invert if background is bright
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find bounding box
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Crop with padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        
        cropped = gray[y1:y2, x1:x2]
        
        # Square canvas
        size = max(cropped.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        offset_x = (size - cropped.shape[1]) // 2
        offset_y = (size - cropped.shape[0]) // 2
        square[offset_y:offset_y+cropped.shape[0], offset_x:offset_x+cropped.shape[1]] = cropped
        
        # Resize to 20x20 then center in 28x28
        resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = resized
    else:
        canvas = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    tensor = canvas.astype(np.float32) / 255.0
    return tensor.reshape(1, 28, 28, 1), canvas

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────
class DigitDetector(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.last_prediction = None
        self.last_confidence = 0
        self.last_update = 0
        self.fps_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
    
    def set_model(self, model):
        self.model = model
    
    def transform(self, frame):
        if self.model is None:
            return frame.to_ndarray(format="bgr24")
        
        img = frame.to_ndarray(format="bgr24")
        
        # FPS calculation
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        # Predict every 500ms (not every frame for performance)
        if current_time - self.last_update > 0.5:
            try:
                # Resize for faster processing
                small = cv2.resize(img, (224, 224))
                tensor, processed = preprocess_frame(small)
                
                predictions = self.model.predict(tensor, verbose=0)[0]
                self.last_prediction = int(np.argmax(predictions))
                self.last_confidence = float(np.max(predictions)) * 100
                self.last_update = current_time
            except:
                pass
        
        # Draw prediction on frame
        if self.last_prediction is not None:
            # Background box
            cv2.rectangle(img, (10, 10), (200, 120), (0, 0, 0), -1)
            cv2.rectangle(img, (10, 10), (200, 120), (30, 144, 255), 2)
            
            # Prediction text
            cv2.putText(img, f"Digit: {self.last_prediction}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(img, f"Conf: {self.last_confidence:.1f}%", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"FPS: {self.fps}", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return img

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">📱 Real-Time Digit Detection</h1>', unsafe_allow_html=True)

# Load model
model, model_path = load_model()

if model is None:
    st.error("❌ Model not found!")
    uploaded = st.file_uploader("Upload .keras model", type=["keras"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
            tmp.write(uploaded.read())
            model = tf.keras.models.load_model(tmp.name)
            st.success("✅ Model loaded!")
            st.rerun()
else:
    st.success(f"✅ Model: `{os.path.basename(model_path)}`")

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")

# Camera settings
st.markdown("### 📷 Camera Settings")
col1, col2, col3 = st.columns(3)

with col1:
    camera_mode = st.selectbox(
        "Camera",
        ["user (Front)", "environment (Back)"],
        index=1
    )
    camera_id = "user" if "Front" in camera_mode else "environment"

with col2:
    resolution = st.selectbox(
        "Resolution",
        ["640x480", "320x240"],
        index=0
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    start_camera = st.button("🎥 Start Camera", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# WEBRTC STREAMER
# ─────────────────────────────────────────────────────────────────────────────
if start_camera or "camera_started" in st.session_state:
    st.session_state.camera_started = True
    
    # Create detector
    detector = DigitDetector()
    detector.set_model(model)
    
    # Camera config
    width, height = map(int, resolution.split("x"))
    
    st.markdown("### 📹 Live Camera Feed")
    
    webrtc_ctx = webrtc_streamer(
        key="digit-detector",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=detector,
        media_stream_constraints={
            "video": {
                "facingMode": camera_id,
                "width": {"ideal": width},
                "height": {"ideal": height},
            },
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    
    if webrtc_ctx.state.playing:
        st.markdown('<p class="status-running">● Camera is running</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-stopped">● Camera stopped</p>', unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # PREDICTION DISPLAY
    # ─────────────────────────────────────────────────────────────────────
    if detector.last_prediction is not None:
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-box">
                <p class="prediction-digit">{detector.last_prediction}</p>
                <p class="prediction-conf">{detector.last_confidence:.1f}% Confidence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Probability bar chart
            try:
                small = cv2.resize(webrtc_ctx.video_transformer.transform.__self__.last_frame 
                                  if hasattr(webrtc_ctx.video_transformer.transform, '__self__') 
                                  else np.zeros((224, 224, 3), dtype=np.uint8), (224, 224))
                tensor, processed = preprocess_frame(small)
                predictions = model.predict(tensor, verbose=0)[0]
                
                # Create bar chart
                prob_data = {f"{i}": float(predictions[i]) * 100 for i in range(10)}
                st.bar_chart(prob_data, height=200)
            except:
                pass
    
    # Stop button
    if st.button("⏹️ Stop Camera"):
        st.session_state.camera_started = False
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD ALTERNATIVE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📷 Alternative: Upload Image")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    tensor, processed = preprocess_frame(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    predictions = model.predict(tensor, verbose=0)[0]
    digit = int(np.argmax(predictions))
    conf = float(np.max(predictions)) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(processed, caption="Processed (28x28)", use_container_width=True)
    
    st.markdown(f"""
    <div class="prediction-box">
        <p class="prediction-digit">{digit}</p>
        <p class="prediction-conf">{conf:.1f}% Confidence</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📱 Real-Time Handwritten Digit Detection | Built with Streamlit + TensorFlow")
