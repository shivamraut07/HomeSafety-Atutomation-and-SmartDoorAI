import cv2
import mediapipe as mp
import os

# ========= CONFIG =========
PERSON_NAME = "Shivam"     # change when adding a new person
CAPTURE_COUNT = 50        # number of NEW images to add
# ==========================

# 🔥 Always resolve path from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SAVE_PATH = os.path.join(DATASET_DIR, PERSON_NAME)

os.makedirs(SAVE_PATH, exist_ok=True)

# Count existing images (APPEND MODE)
existing_images = len([
    f for f in os.listdir(SAVE_PATH)
    if f.lower().endswith(".jpg")
])

start_count = existing_images
target_count = start_count + CAPTURE_COUNT

print(f"[INFO] Saving images to: {SAVE_PATH}")
print(f"[INFO] Existing images: {start_count}")
print(f"[INFO] Capturing until: {target_count}")

# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = start_count
print("[INFO] Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            face = frame[y:y+bh, x:x+bw]

            if face.size != 0 and count < target_count:
                face = cv2.resize(face, (200, 200))
                img_path = os.path.join(
                    SAVE_PATH, f"img_{count+1:03}.jpg"
                )
                cv2.imwrite(img_path, face)
                count += 1
                print(f"[SAVED] {img_path}")

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{count}/{target_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= target_count:
        print("[DONE] Face capture completed")
        break

cap.release()
cv2.destroyAllWindows()
