import cv2
import numpy as np
import os
import urllib.request
import logging

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions, VisionRunningMode

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------
# Download pose_landmarker_lite model if missing
# -----------------------
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

if not os.path.exists(MODEL_PATH):
    logger.info("Downloading pose_landmarker_lite.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Model downloaded successfully")

# -----------------------
# Initialize PoseLandmarker
# -----------------------
options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)
logger.info("PoseLandmarker initialized (Tasks API)")

# -----------------------
# Image analysis function
# -----------------------
def analyze_posture_image(image_bytes: bytes):
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return ["Invalid image uploaded"]

        mp_image = vision.MPImage(
            image_format=vision.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        result = pose_landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return ["No runner detected. Upload full-body running image."]

        lm = result.pose_landmarks[0]
        feedback = []

        # LEFT_SHOULDER = 11, LEFT_HIP = 23, LEFT_KNEE = 25, LEFT_ANKLE = 27, LEFT_HEEL = 29
        shoulder = lm[11]
        hip = lm[23]
        knee = lm[25]
        ankle = lm[27]
        heel = lm[29]

        if shoulder.y > hip.y:
            feedback.append("Leaning forward excessively. Keep upright.")
        else:
            feedback.append("Good upright torso detected.")

        if knee.y < hip.y:
            feedback.append("Good knee lift detected.")
        else:
            feedback.append("Low knee lift detected.")

        if heel.y < ankle.y:
            feedback.append("Possible heel strike detected.")

        return feedback

    except Exception as e:
        logger.exception("Posture image analysis failed")
        return [f"Posture analysis failed: {str(e)}"]
