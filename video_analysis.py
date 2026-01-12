import cv2
import tempfile
import os
import logging
import mediapipe as mp
from posture_tasks import pose_landmarker, vision  # Reuse the model

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------
# Video analysis function
# -----------------------
def analyze_running_video(video_bytes: bytes):
    feedback = set()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        video_path = f.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ["Invalid video uploaded"]

    try:
        frame_count = 0
        MAX_FRAMES = 60

        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = vision.MPImage(
                image_format=vision.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = pose_landmarker.detect(mp_image)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                heel = lm[29]  # LEFT_HEEL
                toe = lm[31]   # LEFT_FOOT_INDEX

                if heel.y < toe.y:
                    feedback.add("Possible heel striking detected.")

            frame_count += 1

        if not feedback:
            feedback.add("Running form appears consistent.")

        return list(feedback)

    except Exception as e:
        logger.exception("Running video analysis failed")
        return [f"Video analysis failed: {str(e)}"]

    finally:
        cap.release()
        os.remove(video_path)
