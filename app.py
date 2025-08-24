import streamlit as st
st.set_page_config(
    page_title="🛡️ SIGAP - Smart Identification and Guarding Alert Platform",
    layout="wide"
)

import tensorflow as tf
import cv2
import numpy as np
import tempfile
import os
import time
import pickle
from ultralytics import YOLO

PYGAME_AVAILABLE = False
PYTTSX3_AVAILABLE = False
WINSOUND_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    pass

def create_model_architecture():
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            input_shape=(16, 224, 224, 3)  # Make sure it's integer
        ),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((2, 2))
        ),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        ),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(10, activation='softmax')  # Adjust output classes
    ])
    return model
    
@st.cache_resource
def load_action_model():
    # tf.compat.v1.disable_eager_execution()
    try:
        model = tf.keras.models.load_model("model.h5", compile=False)
        st.success("✅ Crime detection model loaded")
        return model
    except:
        try:
            model = create_model_architecture()
            model.load_weights('crime_model.h5')
            return model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return None
    # except Exception as e:
    #     st.warning(f"Crime model not found: {e}")
    #     return None
    # finally:
    #     tf.compat.v1.enable_eager_execution()

@st.cache_resource
def load_face_model():
    try:
        model = YOLO("yolov8m.pt")  
        st.success("✅ YOLOv8 face detection model loaded")
        return model
    except Exception as e:
        st.warning(f"Face detection model not found: {e}")
        return None

action_model = load_action_model()
face_model = load_face_model()
crime_classes = ["Burglary", "Fighting", "Normal_Videos_for_Event_Recognition", "Shooting"]
unique_faces = []  

def is_new_face(face_crop, threshold=0.8):
    if len(unique_faces) == 0:
        return True
    face_resized = cv2.resize(face_crop, (64,64)).flatten().astype(np.float32)
    face_resized /= np.linalg.norm(face_resized) + 1e-6
    for uf in unique_faces:
        sim = np.dot(face_resized, uf)
        if sim > threshold:
            return False
    return True

def register_face(face_crop):
    face_resized = cv2.resize(face_crop, (64,64)).flatten().astype(np.float32)
    face_resized /= np.linalg.norm(face_resized) + 1e-6
    unique_faces.append(face_resized)

def play_alarm_sound(alarm_type="crime"):
    if not hasattr(st.session_state, 'alarm_enabled') or not st.session_state.alarm_enabled:
        return
    
    try:
        if WINSOUND_AVAILABLE:
            frequency_map = {
                "crime": 1000,
                "unknown_person": 800, 
                "intruder": 1200,
                "emergency": 1500
            }
            freq = frequency_map.get(alarm_type, 1000)
            winsound.Beep(freq, 500)
            
    except Exception as e:
        pass

def speak_alert(message):
    if not hasattr(st.session_state, 'voice_alerts') or not st.session_state.voice_alerts:
        return
    
    try:
        if PYTTSX3_AVAILABLE:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(message)
            engine.runAndWait()
    except Exception as e:
        pass

if 'alerts_log' not in st.session_state:
    st.session_state.alerts_log = []
if 'alarm_enabled' not in st.session_state:
    st.session_state.alarm_enabled = True
if 'voice_alerts' not in st.session_state:
    st.session_state.voice_alerts = True

st.title("🛡️ SIGAP - Smart Identification and Guarding Alert Platform")
st.markdown("### 🎯 Real-time Crime Detection + Face Recognition")

crime_threshold = 0.7
face_confidence = 0.5
frame_skip = 3
max_faces_display = 20
show_confidence = True
enhanced_overlay = True

st.markdown("---")
st.subheader("📹 Upload Security Video")

uploaded_file = st.file_uploader(
    "Select video file", 
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload CCTV footage for analysis"
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    temp_path = tfile.name

    st.success("🔴 SECURITY MONITORING ACTIVE")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🔴 LIVE FEED")
        video_placeholder = st.empty()
        st.subheader("👤 Faces Detected (Unique)")
        faces_placeholder = st.empty()

    with col2:
        st.subheader("🚨 ALERTS")
        alert_placeholder = st.empty()
        st.subheader("📊 STATUS")
        status_placeholder = st.empty()

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        st.error("Cannot open video file.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        frames_buf = []
        window_size = 16
        frame_count = 0
        alert_count = 0
        
        prediction_interval = 8
        face_infer_interval = 2
        ui_update_interval = 2
        
        detection_status = "🟢 NORMAL"
        alert_level = "normal"
        confidence = 0.0
        label = "Normal_Videos_for_Event_Recognition"
        
        faces_gallery = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps
            timestamp = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
            progress = frame_count / total_frames
        
            if frame_count % frame_skip != 0:
                continue

            if action_model is not None:
                frame_resized = cv2.resize(frame, (224, 224))
                frames_buf.append(frame_resized)

                if len(frames_buf) >= window_size and (len(frames_buf) - window_size) % prediction_interval == 0:
                    try:
                        clip = np.expand_dims(np.array(frames_buf[-window_size:]), axis=0)
                        clip = clip.astype(np.float32) / 255.0  
                        preds = action_model.predict(clip, verbose=0)[0]
                        max_idx = int(np.argmax(preds))
                        label = crime_classes[max_idx]
                        confidence = float(preds[max_idx])

                        if label != "Normal_Videos_for_Event_Recognition":
                            if confidence >= crime_threshold:
                                detection_status = f"🔴 {label.upper()} DETECTED!"
                                alert_level = "critical"
                                alert_count += 1
                                
                                play_alarm_sound("crime")
                                speak_alert(f"Critical alert: {label} detected")
                            elif confidence >= 0.5:
                                detection_status = f"🟡 {label} (Warning)"
                                alert_level = "warning"
                                play_alarm_sound("suspicious")
                            else:
                                detection_status = "🟢 NORMAL"
                                alert_level = "normal"
                        else:
                            detection_status = "🟢 NORMAL"
                            alert_level = "normal"

                        if len(frames_buf) > window_size:
                            frames_buf.pop(0)
                            
                    except Exception as e:
                        st.error(f"Action detection error: {e}")

            if face_model is not None and alert_level in ["critical", "warning"] and frame_count % face_infer_interval == 0:
                try:
                    results = face_model.predict(frame, conf=face_confidence, verbose=False)
                    if len(results) > 0:
                        r = results[0]
                        boxes = r.boxes
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                cls = int(box.cls[0]) if box.cls is not None else -1
                                if cls != 0:   
                                    continue

                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                x1 = max(0, min(x1, frame.shape[1]-1))
                                x2 = max(0, min(x2, frame.shape[1]-1))
                                y1 = max(0, min(y1, frame.shape[0]-1))
                                y2 = max(0, min(y2, frame.shape[0]-1))

                                if x2 > x1 and y2 > y1:
                                    crop = frame[y1:y2, x1:x2].copy()
                                    if is_new_face(crop):
                                        register_face(crop)
                                        faces_gallery.append(crop)
                                        if len(faces_gallery) > max_faces_display:
                                            faces_gallery.pop(0)
                                        play_alarm_sound("unknown_person")
                                        speak_alert("Unknown person detected during security alert")
                                        
                except Exception as e:
                    st.error(f"Face detection error: {e}")

            if frame_count % ui_update_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, caption="Live Security Feed", use_container_width=True)

                if alert_level in ["critical", "warning"] and faces_gallery:
                    n = len(faces_gallery)
                    cols = min(4, n)
                    if cols > 0:
                        face_cols = st.columns(cols)
                        for i, face in enumerate(faces_gallery[-cols:]):  
                            with face_cols[i]:
                                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                st.image(face_rgb, caption=f"Unknown #{i+1}", width=120)
                else:
                    faces_placeholder.empty()

                timestamp = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
                progress = frame_count / max(total_frames, 1)

                alert_html = f"""
                <div style='background: linear-gradient(135deg, 
                    {"#ff1744" if alert_level=="critical" else "#ff9800" if alert_level=="warning" else "#4caf50"}, 
                    {"#ff5722" if alert_level=="critical" else "#ffc107" if alert_level=="warning" else "#8bc34a"}); 
                    padding: 15px; border-radius: 10px; margin: 5px 0; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
                    <h4 style="margin: 0; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{detection_status}</h4>
                    <p style="margin: 4px 0 0 0; color: white;"><strong>Time:</strong> {timestamp}</p>
                    <p style="margin: 4px 0 0 0; color: white;"><strong>Frame:</strong> {frame_count}</p>
                    <p style="margin: 4px 0 0 0; color: white;"><strong>Type:</strong> {label}</p>
                    <p style="margin: 4px 0 0 0; color: white;"><strong>Confidence:</strong> {confidence:.3f}</p>
                    <p style="margin: 4px 0 0 0; color: white;"><strong>Faces:</strong> {len(faces_gallery)}</p>
                </div>
                """
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

                status_html = f"""
                <div style='background-color: #f0f0f0; padding: 15px; border-radius: 10px; 
                           box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <p style="margin: 0;"><strong>Progress:</strong> {progress*100:.1f}%</p>
                    <p style="margin: 8px 0 0 0;"><strong>Total Alerts:</strong> {alert_count}</p>
                    <p style="margin: 8px 0 0 0;"><strong>Unique Faces:</strong> {len(unique_faces)}</p>
                    <p style="margin: 8px 0 0 0;"><strong>Status:</strong> 
                       {"🔴 HIGH ALERT" if alert_level=="critical" else "🟡 WARNING" if alert_level=="warning" else "🟢 SECURE"}</p>
                    <p style="margin: 8px 0 0 0; font-size: 12px;"><strong>Models:</strong> Action + Face AI</p>
                </div>
                """
                status_placeholder.markdown(status_html, unsafe_allow_html=True)

            time.sleep(0.005)

        cap.release()
        
        st.success("🏁 Security monitoring completed successfully!")
        
        st.subheader("📋 Security Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚨 Crime Alerts", alert_count)
        with col2:
            st.metric("👥 Unique Faces", len(unique_faces))
        with col3:
            st.metric("⏱️ Duration", f"{int(total_frames/fps//60):02d}:{int(total_frames/fps%60):02d}")
        with col4:
            st.metric("📊 Frames Processed", frame_count)
        
        if alert_count > 0 or len(unique_faces) > 0:
            st.subheader("🔍 Detailed Analysis")
            if alert_count > 0:
                st.error(f"**{alert_count}** security incidents detected")
                st.write("⚠️ **Action Required:** Review flagged incidents and consider security measures")
            if len(unique_faces) > 0:
                st.warning(f"**{len(unique_faces)}** unique individuals detected during alerts")
                st.write("📋 **Recommendation:** Cross-reference with authorized personnel database")

    try:
        os.remove(temp_path)
    except:
        pass

else:

    st.info("👆 Upload a video file to start security monitoring")




