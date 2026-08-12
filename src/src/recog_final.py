# recognize_live_stream.py - COMPLETE FIXED VERSION
import paho.mqtt.client as mqtt
import json
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from collections import deque
import threading
import socket

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODING_PATH = os.path.join(BASE_DIR, "encodings", "faces.pkl")

CONFIDENCE_THRESHOLD = 0.75      
UNKNOWN_PERSON_THRESHOLD = 0.65  
REQUIRED_MATCHES = 6          
PUBLISH_COOLDOWN = 3            

# ALLOWED and BLOCKED lists
ALLOWED_PERSONS = ["Shivam", "Friend1", "Family"]  # Add your name here
BLOCKED_PERSONS = ["Intruder", "Thief"]  # Add blocked names here
# ==========================================

# ================= MQTT CONFIG =================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "smartdoor/face"
CLIENT_ID = "face_ai_publisher"
# ===============================================

# ================= STREAMING CONFIG =================
STREAM_PORT = 8080
# ====================================================

# Global variables
current_frame = None
frame_lock = threading.Lock()
last_publish_time = 0

current_status = {
    "status": "waiting",
    "face": "None",
    "confidence": 0,
    "timestamp": time.time()
}

# ============= FIXED MJPEG STREAMER =============
class MJPEGStreamer:
    def __init__(self, port=8080):
        self.port = port
        self.running = True
        
    def start(self):
        thread = threading.Thread(target=self._serve, daemon=True)
        thread.start()
        print(f"✅ MJPEG Streamer: http://localhost:{self.port}/stream")
    
    def _serve(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(5)
        print(f"📡 Streaming server listening on port {self.port}")
        
        while self.running:
            try:
                client, addr = server.accept()
                thread = threading.Thread(target=self._handle, args=(client,))
                thread.daemon = True
                thread.start()
            except:
                pass
    
    def _handle(self, client):
        try:
            request = client.recv(1024).decode()
            if 'GET /stream' in request or 'GET /' in request:
                client.send(b'HTTP/1.1 200 OK\r\n')
                client.send(b'Content-Type: multipart/x-mixed-replace; boundary=frame\r\n')
                client.send(b'Cache-Control: no-cache\r\n')
                client.send(b'\r\n')
                
                while self.running:
                    global current_frame
                    with frame_lock:
                        if current_frame is not None:
                            ret, buffer = cv2.imencode('.jpg', current_frame, 
                                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if ret:
                                frame_data = buffer.tobytes()
                                client.send(b'--frame\r\n')
                                client.send(b'Content-Type: image/jpeg\r\n')
                                client.send(f'Content-Length: {len(frame_data)}\r\n'.encode())
                                client.send(b'\r\n')
                                client.send(frame_data)
                                client.send(b'\r\n')
                    time.sleep(0.04)
        except:
            pass
        finally:
            client.close()

# Start streamer
streamer = MJPEGStreamer(port=STREAM_PORT)
streamer.start()
# ====================================================

print("[DEBUG] Encoding path:", ENCODING_PATH)

if not os.path.exists(ENCODING_PATH):
    print("❌ faces.pkl not found. Train first.")
    exit()

with open(ENCODING_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"[INFO] Loaded {len(known_encodings)} face encodings")
print(f"[INFO] Known people: {set(known_names)}")
print(f"[INFO] ALLOWED: {ALLOWED_PERSONS}")
print(f"[INFO] BLOCKED: {BLOCKED_PERSONS}")

# Mediapipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

# MQTT SETUP
try:
    mqtt_client = mqtt.Client(client_id=CLIENT_ID)
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("✅ MQTT connected")
except Exception as e:
    print(f"⚠️ MQTT connection failed: {e}")

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

if not cap.isOpened():
    print("❌ Camera failed to open")
    exit()

print("✅ Camera opened")
print("🚀 Face recognition system ready!")
print("📹 Stream URL: http://localhost:8080/stream")
print("[INFO] Press Q to quit")
print("="*50)

match_history = deque(maxlen=REQUIRED_MATCHES)

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def find_best_match(face_vec, known_encodings, known_names):
    best_score = -1
    best_name = "Unknown"
    
    for i, enc in enumerate(known_encodings):
        score = cosine_similarity(face_vec, enc)
        if score > best_score:
            best_score = score
            best_name = known_names[i]
    
    if best_score < UNKNOWN_PERSON_THRESHOLD:
        return "Unknown", best_score, "unknown"
    
    if best_name in ALLOWED_PERSONS:
        return best_name, best_score, "allowed"
    elif best_name in BLOCKED_PERSONS:
        return best_name, best_score, "blocked"
    else:
        return best_name, best_score, "unknown"

def publish_mqtt(status, name, confidence, category):
    global current_status, last_publish_time
    
    now = time.time()
    if now - last_publish_time < PUBLISH_COOLDOWN:
        return
    
    last_publish_time = now
    
    payload = {
        "status": status,
        "name": name,
        "confidence": round(confidence, 3),
        "category": category,
        "timestamp": now
    }
    
    current_status = payload
    
    try:
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        print(f"📤 MQTT: {status} - {name} ({category}) [{confidence:.2f}]")
    except Exception as e:
        print(f"❌ MQTT Publish Error: {e}")

def preprocess_face(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    face_vec = gray.flatten().astype("float32") / 255.0
    return face_vec

# ================= MAIN LOOP =================
try:
    consecutive_unknown = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                bw = min(w - x, bw + 2*padding)
                bh = min(h - y, bh + 2*padding)
                
                face = frame[y:y+bh, x:x+bw]

                if face.size == 0:
                    continue

                face_vec = preprocess_face(face)
                best_name, best_score, category = find_best_match(face_vec, known_encodings, known_names)
                
                # Decision logic
                if category == "allowed" and best_score > CONFIDENCE_THRESHOLD:
                    match_history.append(best_name)
                    current_face = best_name
                    consecutive_unknown = 0
                    
                    if match_history.count(current_face) >= REQUIRED_MATCHES:
                        label = f"✓ ACCESS GRANTED: {current_face}"
                        color = (0, 255, 0)
                        publish_mqtt("access_granted", current_face, best_score, "allowed")
                    else:
                        label = f"⏳ VERIFYING: {current_face} ({len(match_history)}/{REQUIRED_MATCHES})"
                        color = (255, 255, 0)
                        
                elif category == "blocked":
                    match_history.clear()
                    consecutive_unknown += 1
                    label = "🚫 BLOCKED - ACCESS DENIED"
                    color = (128, 0, 128)
                    publish_mqtt("access_denied", best_name, best_score, "blocked")
                    
                else:
                    match_history.clear()
                    consecutive_unknown += 1
                    
                    if consecutive_unknown > 3:
                        label = "🚨 INTRUDER DETECTED"
                        color = (0, 0, 255)
                        publish_mqtt("intruder_detected", "Unknown", best_score, "unknown")
                    else:
                        label = "⏳ ANALYZING..."
                        color = (0, 120, 255)

                # Draw on frame
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 3)
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Conf: {best_score:.2f} [{category}]", (x, y+bh+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            match_history.clear()
            consecutive_unknown = 0

        # 🔥 CRITICAL: Update global frame with ALL drawings
        with frame_lock:
            current_frame = frame.copy()

        # Show status on local window
        cv2.putText(frame, f"ALLOWED: {ALLOWED_PERSONS}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"BLOCKED: {BLOCKED_PERSONS}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)

        cv2.imshow("Smart Door AI - Access Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n👋 Shutting down...")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    except:
        pass
    streamer.running = False
    print("✅ System shutdown complete")