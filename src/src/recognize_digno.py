# recognize_live_diagnostic.py - SHOWS SCORES WITHOUT REJECTING
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODING_PATH = os.path.join(BASE_DIR, "encodings", "faces.pkl")

with open(ENCODING_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def preprocess_face(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = gray.flatten().astype("float32") / 255.0
    return gray

print("\n" + "="*50)
print("🔍 FACE RECOGNITION DIAGNOSTIC MODE")
print("="*50)
print("Show your face to the camera")
print("Press 'q' to quit")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            x, y = max(0, x), max(0, y)
            face = frame[y:y+bh, x:x+bw]
            
            if face.size > 0:
                face_vec = preprocess_face(face)
                
                # Calculate scores for ALL known faces
                print("\n📊 SCORES:")
                best_score = -1
                best_name = "Unknown"
                
                for i, enc in enumerate(known_encodings):
                    score = cosine_similarity(face_vec, enc)
                    name = known_names[i]
                    print(f"  {name}: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_name = name
                
                # Draw with score
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_name}: {best_score:.4f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Diagnostic - Find Your Score", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()