import cv2
import numpy as np
import threading
import time

# ------------------------
# Video Capture Buffer Class
# ------------------------
class VideoCaptureBuffer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.buffer_frame = None
        self.stopped = False
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.buffer_frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = self.buffer_frame
        return frame is not None, frame

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

cap = VideoCaptureBuffer(r"D:\camera_tamper\video_8.avi")

# ------------------------
# Parameters (DAYTIME - UNCHANGED)
# ------------------------
FACE_AREA_RATIO = 0.25
LOW_STD_THRESH = 18
EDGE_DENSITY_THRESH = 0.02
ALERT_FRAMES = 20

# ------------------------
# Night detection params
# ------------------------
NIGHT_BRIGHTNESS_THRESH = 60   # tune once if needed

tamper_count = 0

# ------------------------
# Main loop
# ------------------------
cv2.namedWindow("Camera Tampering Detection", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    frame_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------------
    # Night detection
    # ------------------------
    mean_brightness = np.mean(gray)
    is_night = mean_brightness < NIGHT_BRIGHTNESS_THRESH

    # ------------------------
    # 2️⃣ Hand / cloth cover detection
    # ------------------------
    stddev = np.std(gray)
    low_contrast_alert = stddev < LOW_STD_THRESH

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / frame_area
    low_texture_alert = edge_density < EDGE_DENSITY_THRESH

    # Day logic UNCHANGED, Night logic guarded
    if is_night:
        # Require extreme uniformity to trigger at night
        hand_cover_alert = low_contrast_alert and low_texture_alert and stddev < 8
    else:
        hand_cover_alert = low_contrast_alert and low_texture_alert

    # ------------------------
    # Final decision
    # ------------------------
    if hand_cover_alert or edge_density<0.02:  #
        
        tamper_count += 1
    else:
        tamper_count = max(0, tamper_count - 1)
    # print("tamper_count",tamper_count)
    if tamper_count >= ALERT_FRAMES:
        cv2.putText(frame, "CAMERA TAMPERING DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # ------------------------
    # Debug info
    # ------------------------
    cv2.putText(frame, f"STD: {stddev:.1f}",
                (10, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 10)

    cv2.putText(frame, f"Edge density: {edge_density:.3f}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 10)

    cv2.imshow("Camera Tampering Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------------
# Cleanup
# ------------------------
cap.release()
cv2.destroyAllWindows()
