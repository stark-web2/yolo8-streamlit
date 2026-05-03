import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading
import time
import os
from datetime import datetime
from collections import defaultdict

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Live Object Detection & Tracking", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .alert-box {
        background: #ff000022;
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 12px;
        color: #ff4444;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .saved-frame-info {
        background: #00ff0011;
        border: 1px solid #44ff44;
        border-radius: 8px;
        padding: 10px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 Live Object Detection & Tracking")
st.write("Turn on your webcam to detect, count, alert, and save detected objects in real time.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# SHARED STATE (thread-safe)
# -----------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {}           # {class_name: count}
        self.alert_triggered = {}  # {class_name: True/False}
        self.saved_frames = []     # list of saved file paths
        self.last_frame = None     # latest annotated frame (numpy)

state = SharedState()

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Settings")

# Alert configuration
st.sidebar.subheader("🚨 Alert Triggers")
alert_objects = st.sidebar.multiselect(
    "Alert when these objects are detected:",
    options=[
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "bottle", "phone", "laptop", "knife", "scissors"
    ],
    default=["person"]
)

alert_threshold = st.sidebar.slider(
    "Alert if count exceeds:",
    min_value=1, max_value=20, value=3
)

# Save frame configuration
st.sidebar.subheader("💾 Frame Saving")
save_objects = st.sidebar.multiselect(
    "Auto-save frames when detected:",
    options=[
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "bottle", "phone", "laptop", "knife", "scissors"
    ],
    default=[]
)

save_cooldown = st.sidebar.slider(
    "Save cooldown (seconds):",
    min_value=1, max_value=30, value=5,
    help="Minimum seconds between saves for the same object"
)

save_dir = "detected_frames"
os.makedirs(save_dir, exist_ok=True)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Confidence threshold:", 0.1, 1.0, 0.4, 0.05
)

# -----------------------------
# VIDEO PROCESSOR CLASS
# -----------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.last_save_time = defaultdict(float)  # {class_name: timestamp}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO tracking
        results = self.model.track(img, persist=True, verbose=False, conf=conf_threshold)
        annotated_frame = results[0].plot()

        # --- Count objects ---
        counts = defaultdict(int)
        boxes = results[0].boxes
        if boxes is not None and boxes.cls is not None:
            for cls_id in boxes.cls.tolist():
                class_name = self.model.names[int(cls_id)]
                counts[class_name] += 1

        # --- Check alerts ---
        alert_status = {}
        for obj in alert_objects:
            count = counts.get(obj, 0)
            alert_status[obj] = count >= alert_threshold

        # --- Auto-save frames ---
        now = time.time()
        saved_paths = []
        for obj in save_objects:
            if counts.get(obj, 0) > 0:
                elapsed = now - self.last_save_time[obj]
                if elapsed >= save_cooldown:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{save_dir}/{obj}_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    self.last_save_time[obj] = now
                    saved_paths.append(filename)

        # --- Update shared state ---
        with state.lock:
            state.counts = dict(counts)
            state.alert_triggered = alert_status
            state.last_frame = annotated_frame.copy()
            if saved_paths:
                state.saved_frames.extend(saved_paths)
                state.saved_frames = state.saved_frames[-20:]  # keep last 20

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -----------------------------
# LAYOUT: VIDEO + STATS
# -----------------------------
col_video, col_stats = st.columns([2, 1])

with col_video:
    webrtc_streamer(
        key="yolo-live",
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_stats:
    st.subheader("📊 Live Object Counts")
    counts_placeholder = st.empty()

    st.subheader("🚨 Alerts")
    alerts_placeholder = st.empty()

    st.subheader("💾 Saved Frames")
    saves_placeholder = st.empty()

# -----------------------------
# MANUAL SAVE BUTTON
# -----------------------------
st.divider()
if st.button("📸 Save Current Frame Manually"):
    with state.lock:
        frame = state.last_frame
    if frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_dir}/manual_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        with state.lock:
            state.saved_frames.append(filename)
        st.success(f"Frame saved: `{filename}`")
    else:
        st.warning("No frame available yet. Start the webcam first.")

# -----------------------------
# REFRESH STATS LOOP
# -----------------------------
refresh_rate = st.sidebar.slider("UI refresh rate (sec):", 0.5, 3.0, 1.0, 0.5)

stats_area = st.empty()

while True:
    with state.lock:
        counts = dict(state.counts)
        alerts = dict(state.alert_triggered)
        saved = list(state.saved_frames)

    # --- Counts display ---
    if counts:
        counts_md = ""
        for obj, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt, 20)
            counts_md += f"**{obj}** — `{cnt}`  \n{bar}\n\n"
        counts_placeholder.markdown(counts_md)
    else:
        counts_placeholder.info("No objects detected yet.")

    # --- Alerts display ---
    alert_html = ""
    for obj, triggered in alerts.items():
        if triggered:
            alert_html += f'<div class="alert-box">⚠️ {obj.upper()} count ≥ {alert_threshold}!</div>'
    if alert_html:
        alerts_placeholder.markdown(alert_html, unsafe_allow_html=True)
    else:
        alerts_placeholder.success("No alerts triggered.")

    # --- Saved frames display ---
    if saved:
        saves_md = "\n".join([f"- `{os.path.basename(p)}`" for p in reversed(saved[-5:])])
        saves_placeholder.markdown(
            f'<div class="saved-frame-info">📁 Last {min(5, len(saved))} saved:<br>{saves_md}</div>',
            unsafe_allow_html=True
        )
    else:
        saves_placeholder.info("No frames saved yet.")

    time.sleep(refresh_rate)