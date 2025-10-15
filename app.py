# <<<<<<< HEAD
import streamlit as st
import tempfile
import os
import cv2
import time
import smtplib
import sqlite3
from ultralytics import YOLO
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================
# 📄 Streamlit Page Config
# ===============================
st.set_page_config(page_title="AI-ML CCTV Prototype", page_icon="📹", layout="wide")
st.title("🎯 AI & ML CCTV Surveillance System with Alerts & Logging")

# ===============================
# ⚙️ Sidebar Config
# ===============================
st.sidebar.header("Settings ⚙️")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["YOLOv8n (Fast, Light)", "YOLOv8s (More Accurate)"]
)
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)

alert_objects = st.sidebar.multiselect(
    "🚨 Objects to Trigger Alerts",
    ["person", "car", "knife", "gun", "bottle", "cell phone"],
    default=["person", "car"]
)

enable_email_alert = st.sidebar.checkbox("📧 Enable Email Alerts")
receiver_email = None
if enable_email_alert:
    receiver_email = st.sidebar.text_input("Enter recipient email address")

# ===============================
# 🧠 Load YOLO model
# ===============================
model_name = "yolov8n.pt" if "n" in model_choice.lower() else "yolov8s.pt"
model = YOLO(model_name)

# ===============================
# 💾 Setup SQLite Database
# ===============================
conn = sqlite3.connect(os.path.join(tempfile.gettempdir(), "detections.db"))
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source TEXT,
    object TEXT,
    confidence REAL
)
""")
conn.commit()

# ===============================
# 📧 Email Alert Function
# ===============================
def send_email_alert(label, conf):
    sender_email = "your_email@gmail.com"     # ⚠️ Replace with your email
    app_password = "your_app_password"        # ⚠️ Replace with an App Password
    subject = f"🚨 ALERT: {label.upper()} Detected!"
    body = f"A {label.upper()} was detected with confidence {conf*100:.1f}%"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)
    except Exception as e:
        st.error(f"Email alert failed: {e}")

# ===============================
# 💾 Save alert snapshot
# ===============================
def save_alert_snapshot(frame, label):
    alerts_dir = os.path.join(tempfile.gettempdir(), "alerts")
    os.makedirs(alerts_dir, exist_ok=True)
    filename = f"{label}_{int(time.time())}.jpg"
    filepath = os.path.join(alerts_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath

# ===============================
# 🧾 Log detection
# ===============================
def log_detection(source, label, conf):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO detections (timestamp, source, object, confidence) VALUES (?, ?, ?, ?)",
              (timestamp, source, label, conf))
    conn.commit()

# ===============================
# 🎥 Source Selection
# ===============================
option = st.radio("Choose Input Source",
                  ("📤 Upload Video", "📷 Webcam", "🛰️ CCTV / IP Camera"),
                  horizontal=True)

# ===============================
# 📤 Upload Video
# ===============================
if option == "📤 Upload Video":
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)
        st.info("Processing video... ⏳")

        results = model.predict(
            source=tfile.name,
            conf=confidence,
            save=True,
            project=tempfile.gettempdir(),
            name="output",
            exist_ok=True,
            stream=True
        )

        counts = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf_val = float(box.conf[0])
                counts[label] = counts.get(label, 0) + 1
                log_detection("uploaded_video", label, conf_val)

        st.success("✅ Video processed.")
        df = pd.DataFrame(list(counts.items()), columns=["Object", "Count"])
        st.dataframe(df)
        csv_path = os.path.join(tempfile.gettempdir(), "summary.csv")
        df.to_csv(csv_path, index=False)
        st.download_button("⬇️ Download Summary", open(csv_path, "rb"), file_name="summary.csv")

# ===============================
# 📷 Webcam Live Detection
# ===============================
elif option == "📷 Webcam":
    st.subheader("🎥 Live Webcam Feed")
    run_webcam = st.checkbox("Start Webcam", key="webcam_run")
    frame_placeholder = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        while st.session_state.get("webcam_run"):
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Webcam not accessible.")
                break

            results = model(frame, conf=confidence)
            annotated = results[0].plot()

            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                conf_val = float(box.conf[0])
                if label in alert_objects and conf_val >= confidence:
                    save_alert_snapshot(frame, label)
                    log_detection("webcam", label, conf_val)
                    st.warning(f"🚨 ALERT: {label.upper()} Detected!")
                    if enable_email_alert and receiver_email:
                        send_email_alert(label, conf_val)

            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.03)
        cap.release()

# ===============================
# 🛰️ CCTV / IP Camera Stream
# ===============================
elif option == "🛰️ CCTV / IP Camera":
    st.subheader("🛰️ Live CCTV Stream (RTSP / HTTP)")
    cctv_url = st.text_input("Enter CCTV Stream URL", placeholder="e.g., rtsp://admin:1234@192.168.1.10:554/stream1")
    start_cctv = st.checkbox("Start CCTV", key="cctv_run")
    frame_placeholder = st.empty()

    if start_cctv and cctv_url:
        cap = cv2.VideoCapture(cctv_url)
        if not cap.isOpened():
            st.error("❌ Cannot connect to CCTV camera.")
        else:
            while st.session_state.get("cctv_run"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Stream interrupted.")
                    break

                results = model(frame, conf=confidence)
                annotated = results[0].plot()

                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    conf_val = float(box.conf[0])
                    if label in alert_objects and conf_val >= confidence:
                        save_alert_snapshot(frame, label)
                        log_detection("CCTV", label, conf_val)
                        st.warning(f"🚨 ALERT: {label.upper()} Detected!")
                        if enable_email_alert and receiver_email:
                            send_email_alert(label, conf_val)

                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.03)
            cap.release()

# ===============================
# 📊 Detection Logs
# ===============================
st.markdown("---")
st.subheader("🧾 Detection Logs")
if st.button("Refresh Logs"):
    df_logs = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC LIMIT 20", conn)
    st.dataframe(df_logs)
# =======
import streamlit as st
import tempfile
import os
import cv2
import time
import smtplib
import sqlite3
from ultralytics import YOLO
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ===============================
# 📄 Streamlit Page Config
# ===============================
st.set_page_config(page_title="AI-ML CCTV Prototype", page_icon="📹", layout="wide")
st.title("🎯 AI & ML CCTV Surveillance System with Alerts & Logging")

# ===============================
# ⚙️ Sidebar Config
# ===============================
st.sidebar.header("Settings ⚙️")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["YOLOv8n (Fast, Light)", "YOLOv8s (More Accurate)"]
)
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)

alert_objects = st.sidebar.multiselect(
    "🚨 Objects to Trigger Alerts",
    ["person", "car", "knife", "gun", "bottle", "cell phone"],
    default=["person", "car"]
)

enable_email_alert = st.sidebar.checkbox("📧 Enable Email Alerts")
receiver_email = None
if enable_email_alert:
    receiver_email = st.sidebar.text_input("Enter recipient email address")

# ===============================
# 🧠 Load YOLO model
# ===============================
model_name = "yolov8n.pt" if "n" in model_choice.lower() else "yolov8s.pt"
model = YOLO(model_name)

# ===============================
# 💾 Setup SQLite Database
# ===============================
conn = sqlite3.connect(os.path.join(tempfile.gettempdir(), "detections.db"))
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    source TEXT,
    object TEXT,
    confidence REAL
)
""")
conn.commit()

# ===============================
# 📧 Email Alert Function
# ===============================
def send_email_alert(label, conf):
    sender_email = "your_email@gmail.com"     # ⚠️ Replace with your email
    app_password = "your_app_password"        # ⚠️ Replace with an App Password
    subject = f"🚨 ALERT: {label.upper()} Detected!"
    body = f"A {label.upper()} was detected with confidence {conf*100:.1f}%"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(msg)
    except Exception as e:
        st.error(f"Email alert failed: {e}")

# ===============================
# 💾 Save alert snapshot
# ===============================
def save_alert_snapshot(frame, label):
    alerts_dir = os.path.join(tempfile.gettempdir(), "alerts")
    os.makedirs(alerts_dir, exist_ok=True)
    filename = f"{label}_{int(time.time())}.jpg"
    filepath = os.path.join(alerts_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath

# ===============================
# 🧾 Log detection
# ===============================
def log_detection(source, label, conf):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO detections (timestamp, source, object, confidence) VALUES (?, ?, ?, ?)",
              (timestamp, source, label, conf))
    conn.commit()

# ===============================
# 🎥 Source Selection
# ===============================
option = st.radio("Choose Input Source",
                  ("📤 Upload Video", "📷 Webcam", "🛰️ CCTV / IP Camera"),
                  horizontal=True)

# ===============================
# 📤 Upload Video
# ===============================
if option == "📤 Upload Video":
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)
        st.info("Processing video... ⏳")

        results = model.predict(
            source=tfile.name,
            conf=confidence,
            save=True,
            project=tempfile.gettempdir(),
            name="output",
            exist_ok=True,
            stream=True
        )

        counts = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf_val = float(box.conf[0])
                counts[label] = counts.get(label, 0) + 1
                log_detection("uploaded_video", label, conf_val)

        st.success("✅ Video processed.")
        df = pd.DataFrame(list(counts.items()), columns=["Object", "Count"])
        st.dataframe(df)
        csv_path = os.path.join(tempfile.gettempdir(), "summary.csv")
        df.to_csv(csv_path, index=False)
        st.download_button("⬇️ Download Summary", open(csv_path, "rb"), file_name="summary.csv")

# ===============================
# 📷 Webcam Live Detection
# ===============================
elif option == "📷 Webcam":
    st.subheader("🎥 Live Webcam Feed")
    run_webcam = st.checkbox("Start Webcam", key="webcam_run")
    frame_placeholder = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        while st.session_state.get("webcam_run"):
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Webcam not accessible.")
                break

            results = model(frame, conf=confidence)
            annotated = results[0].plot()

            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                conf_val = float(box.conf[0])
                if label in alert_objects and conf_val >= confidence:
                    save_alert_snapshot(frame, label)
                    log_detection("webcam", label, conf_val)
                    st.warning(f"🚨 ALERT: {label.upper()} Detected!")
                    if enable_email_alert and receiver_email:
                        send_email_alert(label, conf_val)

            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            time.sleep(0.03)
        cap.release()

# ===============================
# 🛰️ CCTV / IP Camera Stream
# ===============================
elif option == "🛰️ CCTV / IP Camera":
    st.subheader("🛰️ Live CCTV Stream (RTSP / HTTP)")
    cctv_url = st.text_input("Enter CCTV Stream URL", placeholder="e.g., rtsp://admin:1234@192.168.1.10:554/stream1")
    start_cctv = st.checkbox("Start CCTV", key="cctv_run")
    frame_placeholder = st.empty()

    if start_cctv and cctv_url:
        cap = cv2.VideoCapture(cctv_url)
        if not cap.isOpened():
            st.error("❌ Cannot connect to CCTV camera.")
        else:
            while st.session_state.get("cctv_run"):
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Stream interrupted.")
                    break

                results = model(frame, conf=confidence)
                annotated = results[0].plot()

                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    conf_val = float(box.conf[0])
                    if label in alert_objects and conf_val >= confidence:
                        save_alert_snapshot(frame, label)
                        log_detection("CCTV", label, conf_val)
                        st.warning(f"🚨 ALERT: {label.upper()} Detected!")
                        if enable_email_alert and receiver_email:
                            send_email_alert(label, conf_val)

                frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(0.03)
            cap.release()

# ===============================
# 📊 Detection Logs
# ===============================
st.markdown("---")
st.subheader("🧾 Detection Logs")
if st.button("Refresh Logs"):
    df_logs = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC LIMIT 20", conn)
    st.dataframe(df_logs)

