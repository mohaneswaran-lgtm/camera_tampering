import cv2
import numpy as np

# ------------------------
# Load face detector
# ------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

FACE_AREA_RATIO = 0.25   # face near camera
LOW_STD_THRESH = 18      # hand/cloth cover
EDGE_DENSITY_THRESH = 0.02
ALERT_FRAMES = 5

tamper_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------
    # 1️⃣ Face detection
    # ------------------------
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5
    )

    face_alert = False
    for (x, y, fw, fh) in faces:
        face_area = fw * fh
        ratio = face_area / frame_area

        if ratio > FACE_AREA_RATIO:
            face_alert = True
            cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,0,255), 2)
            cv2.putText(frame, "FACE TOO CLOSE",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,0,255), 2)

    # ------------------------
    # 2️⃣ Hand / cloth cover detection
    # ------------------------
    stddev = np.std(gray)
    low_contrast_alert = stddev < LOW_STD_THRESH

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / frame_area
    low_texture_alert = edge_density < EDGE_DENSITY_THRESH

    hand_cover_alert = low_contrast_alert and low_texture_alert

    # ------------------------
    # Final decision
    # ------------------------
    if face_alert or hand_cover_alert or edge_density<0.02:
        tamper_count += 1
    else:
        tamper_count = max(0, tamper_count - 1)

    if tamper_count >= ALERT_FRAMES:
        cv2.putText(frame, "🚨 CAMERA TAMPERING 🚨",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 3)

    # ------------------------
    # Debug info
    # ------------------------
    cv2.putText(frame, f"STD: {stddev:.1f}",
                (10, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(frame, f"Edge density: {edge_density:.3f}",
                (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Camera Tampering Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
