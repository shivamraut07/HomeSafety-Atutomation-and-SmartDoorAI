# train_faces_multi.py
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

def extract_features(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.flatten() / 255.0

known_encodings = []
known_names = []

print("="*50)
print("🚀 TRAINING MULTI-PERSON FACE RECOGNITION")
print("="*50)

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue
    
    count = 0
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
                features = extract_features(face)
                known_encodings.append(features)
                known_names.append(person_name)
                count += 1
    
    print(f"  ✅ Added {count} images for '{person_name}'")

# Save encodings
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODING_FILE, "wb") as f:
    pickle.dump(data, f)

print("\n" + "="*50)
print("✅ TRAINING COMPLETE!")
print(f"📊 Total encodings: {len(known_encodings)}")
print(f"👥 People in dataset: {set(known_names)}")
print("="*50)

# Show summary
print("\n📋 DATASET SUMMARY:")
for name in set(known_names):
    count = known_names.count(name)
    print(f"  • {name}: {count} images")