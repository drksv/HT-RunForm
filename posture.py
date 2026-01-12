import cv2
import numpy as np
import os
import urllib.request
import logging
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model setup
# --------------------------------------------------
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

if not os.path.exists(MODEL_PATH):
    logger.info("Pose model not found. Downloading...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Pose model downloaded successfully")

# --------------------------------------------------
# Initialize MediaPipe Pose Landmarker
# --------------------------------------------------
options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)
logger.info("MediaPipe Pose Landmarker initialized")

# --------------------------------------------------
# Main analysis function
# --------------------------------------------------
def analyze_posture_image(image_bytes: bytes):
    """
    Analyze a running posture image and return textual feedback.
    """

    try:
        # Decode image
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("Invalid image received")
            return ["Invalid image uploaded"]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        # Run pose detection
        result = pose_landmarker.detect(mp_image)

        if not result.pose_landmarks:
            logger.info("No pose detected in image")
            return ["No runner detected. Please upload a full-body running image."]

        landmarks = result.pose_landmarks[0]
        feedback = []

        # --------------------------------------------------
        # Posture checks
        # --------------------------------------------------
        shoulder = landmarks[11]  # LEFT_SHOULDER
        hip = landmarks[23]       # LEFT_HIP

        if shoulder.y > hip.y:
            feedback.append(
                "You appear to be leaning forward excessively. Try maintaining a tall, upright running posture."
            )
        else:
            feedback.append("Good upright torso posture detected.")

        # Knee lift
        knee = landmarks[25]  # LEFT_KNEE
        if knee.y < hip.y:
            feedback.append("Good knee lift detected, indicating efficient stride mechanics.")
        else:
            feedback.append("Low knee lift detected. Focus on driving knees slightly higher.")

        # Foot strike estimation
        ankle = landmarks[27]  # LEFT_ANKLE
        heel = landmarks[29]   # LEFT_HEEL

        if heel.y < ankle.y:
            feedback.append(
                "Possible heel strike detected. Consider a mid-foot landing for better efficiency."
            )

        logger.info("Posture analysis completed successfully")
        return feedback

    except Exception as e:
        logger.exception("Posture analysis failed")
        return [f"Posture analysis failed: {str(e)}"]





