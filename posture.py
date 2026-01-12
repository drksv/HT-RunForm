import cv2
import numpy as np
import mediapipe as mp
import logging

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("posture")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"MediaPipe version: {mp.__version__}")

# -------------------------
# Load Pose model
# -------------------------
MODEL_PATH = "pose_landmarker_lite.task"

logger.info("Loading MediaPipe PoseLandmarker model")

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)

pose_landmarker = PoseLandmarker.create_from_options(options)

# -------------------------
# Main analysis function
# -------------------------
def analyze_posture_image(image_bytes: bytes):
    logger.info("Starting posture analysis")

    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("Invalid image uploaded")
            return ["Invalid image uploaded"]

        height, width, _ = img.shape
        logger.info(f"Image loaded: {width}x{height}")

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        )

        result = pose_landmarker.detect(mp_image)

        if not result.pose_landmarks:
            logger.info("No pose detected")
            return [
                "No runner detected. Upload a full-body side-view running image."
            ]

        landmarks = result.pose_landmarks[0]
        feedback = []

        # Helper to get landmark by index
        def lm(idx):
            return landmarks[idx]

        # Indices (MediaPipe spec)
        LEFT_SHOULDER = 11
        LEFT_HIP = 23
        LEFT_KNEE = 25
        LEFT_ANKLE = 27
        LEFT_HEEL = 29

        shoulder = lm(LEFT_SHOULDER)
        hip = lm(LEFT_HIP)
        knee = lm(LEFT_KNEE)
        ankle = lm(LEFT_ANKLE)
        heel = lm(LEFT_HEEL)

        # -------------------------
        # Torso posture
        # -------------------------
        if shoulder.y > hip.y:
            feedback.append(
                "You are leaning forward excessively. Try maintaining an upright torso while running."
            )
            logger.info("Forward lean detected")
        else:
            feedback.append("Good upright torso posture detected.")

        # -------------------------
        # Knee lift
        # -------------------------
        if knee.y < hip.y:
            feedback.append("Good knee lift detected.")
        else:
            feedback.append("Low knee lift detected. Drive knees slightly higher.")

        # -------------------------
        # Foot strike
        # -------------------------
        if heel.y < ankle.y:
            feedback.append(
                "Heel strike pattern detected. Consider a mid-foot landing for efficiency."
            )

        logger.info("Posture analysis completed successfully")
        return feedback

    except Exception as e:
        logger.exception("Posture analysis failed")
        return ["Internal error during posture analysis"]
