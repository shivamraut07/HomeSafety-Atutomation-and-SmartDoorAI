# train_faces_enhanced.py
import cv2
import mediapipe as mp
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ENCODING_FILE = os.path.join(BASE_DIR, "encodings", "faces.pkl")

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

def extract_enhanced_features(face_image):
    """Extract better features for training"""
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Create multiple augmented versions
    features = []
    
    # Original
    features.append(gray.flatten() / 255.0)
    
    # Slightly brighter
    bright = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    features.append(bright.flatten() / 255.0)
    
    # Slightly darker
    dark = cv2.convertScaleAbs(gray, alpha=0.8, beta=-10)
    features.append(dark.flatten() / 255.0)
    
    # Slight rotation (simulated)
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1)
    rotated = cv2.warpAffine(gray, M, (w, h))
    features.append(rotated.flatten() / 255.0)
    
    return features

known_encodings = []
known_names = []

print("🚀 Enhanced Training Started...")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue
    
    print(f"\n📁 Processing: {person_name}")
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        
        if results.detections:
            h, w, _ = image.shape
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            x, y = max(0, x), max(0, y)
            face = image[y:y+bh, x:x+bw]
            
            if face.size > 0:
                # Extract multiple features per face
                features = extract_enhanced_features(face)
                for feat in features:
                    known_encodings.append(feat)
                    known_names.append(person_name)
                
                print(f"  ✅ Added {len(features)} features from {img_name}")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODING_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"\n✅ Training Complete!")
print(f"📊 Total encodings: {len(known_encodings)}")
print(f"👤 People: {set(known_names)}")