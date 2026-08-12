import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()
else:
    print("✅ Camera opened")

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not received")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            bw, bh = int(bbox.width * w), int(bbox.height * h)

            cv2.rectangle(frame, (x,y), (x+bw, y+bh), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
