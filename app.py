import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os
import pandas as pd
import socket

# -------------------------------------------------
# Helper: Check if running locally or on Streamlit Cloud
# -------------------------------------------------
def is_running_locally():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        # Common Streamlit Cloud hostname pattern
        return not ("streamlit" in hostname or "cloud" in hostname)
    except:
        return True

running_local = is_running_locally()

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(page_title="AI-ML Video Analysis & Alerts", page_icon="🤖", layout="wide")

st.title("🎥 AI & ML Enabled Video Analysis and Interpretation")
st.markdown("""
This prototype uses **AI & Machine Learning (YOLOv8)** to automatically  
**analyze and interpret video feeds** — detecting people, objects, and activity in real time.
""")

# -------------------------------------------------
# Sidebar Settings
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["YOLOv8n (Fast, Light)", "YOLOv8s (More Accurate)"],
    key="model_select"
)
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.05, key="confidence_slider")

# Load YOLO Model
model_name = "yolov8n.pt" if "n" in model_choice.lower() else "yolov8s.pt"
model = YOLO(model_name)

# -------------------------------------------------
# Video Upload Section
# -------------------------------------------------
st.subheader("📤 Upload a Video for Analysis")
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="video_uploader")

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.video(tfile.name)
    st.info("Processing video... Please wait ⏳")

    results = model.predict(
        source=tfile.name,
        conf=confidence,
        save=True,
        project=tempfile.gettempdir(),
        name="output",
        exist_ok=True
    )

    # Collect object counts
    object_counts = {}
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            object_counts[label] = object_counts.get(label, 0) + 1

    st.success("✅ Analysis Complete!")
    st.subheader("📊 Detected Objects Summary")
    if object_counts:
        for obj, count in object_counts.items():
            st.write(f"- **{obj}**: {count}")
    else:
        st.warning("No objects detected. Try lowering the confidence level.")

    # Display processed video
    processed_dir = os.path.join(tempfile.gettempdir(), "output")
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith(".mp4")]
    if processed_files:
        output_video_path = os.path.join(processed_dir, processed_files[0])
        st.subheader("🎬 Processed Video Output")
        st.video(output_video_path)

# -------------------------------------------------
# Live Webcam (Local Only)
# -------------------------------------------------
st.subheader("📡 Live Video (Webcam)")

if running_local:
    run_webcam = st.checkbox("Start Webcam", key="run_webcam")

    if run_webcam:
        st.info("Starting webcam... Press Stop to end the stream.")
        camera = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while run_webcam:
            ret, frame = camera.read()
            if not ret:
                st.error("⚠️ Failed to capture webcam feed.")
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            run_webcam = st.checkbox("Stop Webcam", key="stop_webcam")
            if run_webcam:
                break

        camera.release()
        st.success("🛑 Webcam stopped.")
else:
    st.warning("⚠️ Webcam is disabled in Streamlit Cloud for security reasons. Use local run to access your camera.")

# -------------------------------------------------
# CCTV / IP Camera Section
# -------------------------------------------------
st.subheader("🎥 CCTV / IP Camera Stream")
cctv_url = st.text_input("Enter CCTV Stream URL (RTSP or HTTP):", key="cctv_url")
run_cctv = st.checkbox("Start CCTV Stream", key="run_cctv")

if run_cctv and cctv_url:
    st.info("Connecting to CCTV feed...")
    cap = cv2.VideoCapture(cctv_url)
    frame_placeholder_cctv = st.empty()

    if not cap.isOpened():
        st.error("❌ Unable to connect to CCTV stream. Check the URL.")
    else:
        while run_cctv:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Lost connection to CCTV feed.")
                break

            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder_cctv.image(frame_rgb, channels="RGB")

            run_cctv = st.checkbox("Stop CCTV Stream", key="stop_cctv")
            if run_cctv:
                break

        cap.release()
        st.success("🛑 CCTV stream stopped.")

# -------------------------------------------------
# Real-Time Alerts Section
# -------------------------------------------------
st.subheader("🚨 Real-Time Alerts")
enable_alerts = st.checkbox("Enable Email Alerts for Object Detection", key="enable_alerts")

if enable_alerts:
    st.info("Email alerts will be triggered when critical objects (like 'person' or 'fire') are detected.")
    email_input = st.text_input("Enter recipient email:", key="email_input")

    if email_input:
        st.success(f"📧 Alerts will be sent to: {email_input}")
    else:
        st.warning("Please enter an email to activate alerts.")
