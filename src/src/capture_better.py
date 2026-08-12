# capture_better_faces.py
import cv2
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "dataset", "Shivam")
os.makedirs(SAVE_PATH, exist_ok=True)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
count = len([f for f in os.listdir(SAVE_PATH) if f.endswith('.jpg')])

print("📸 CAPTURE BETTER FACE SAMPLES")
print("="*50)
print("✅ Look directly at camera")
print("✅ Different angles: left, right, up, down")
print("✅ Different lighting: bright, normal, dim")
print("✅ Different expressions: smile, serious, neutral")
print("✅ With/without glasses")
print("✅ Press SPACE to capture")
print("✅ Press 'q' to quit")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            x, y = max(0, x), max(0, y)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Capture Better Faces", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar
        if results.detections:
            face = frame[y:y+bh, x:x+bw]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(os.path.join(SAVE_PATH, f"shivam_{count+1:03d}.jpg"), face)
            count += 1
            print(f"✅ Captured {count} images")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Total images: {count}")
print("▶️ Now run train_faces.py again!")