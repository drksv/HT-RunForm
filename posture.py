import cv2
import mediapipe as mp
import numpy as np
import logging

# -------------------------
# Logging configuration
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

# -------------------------
# MediaPipe validation
# -------------------------
logger.info(f"MediaPipe loaded from: {mp.__file__}")
logger.info(f"MediaPipe version: {getattr(mp, '__version__', 'UNKNOWN')}")

if not hasattr(mp, "solutions"):
    logger.error("MediaPipe 'solutions' not found — import shadowing likely.")
    raise RuntimeError("Invalid MediaPipe installation or name collision")

mp_pose = mp.solutions.pose
logger.info("MediaPipe Pose initialized successfully")

# -------------------------
# Main analysis function
# -------------------------
def analyze_posture_image(image_bytes: bytes):
    logger.info("Starting posture analysis")

    try:
        # Convert bytes → NumPy → OpenCV image
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("Uploaded image could not be decoded")
            return ["Invalid image uploaded. Please upload a clear JPG or PNG file."]

        logger.info(f"Image decoded successfully | shape={img.shape}")

        # Run MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:

            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                logger.info("No pose landmarks detected")
                return [
                    "No runner detected. Please upload a full-body running image taken from the side."
                ]

            landmarks = results.pose_landmarks.landmark
            feedback = []

            # -------------------------
            # Torso posture
            # -------------------------
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            if shoulder.y > hip.y:
                feedback.append(
                    "You appear to be leaning forward excessively. Try maintaining a tall, upright running posture."
                )
                logger.info("Detected excessive forward lean")
            else:
                feedback.append("Good upright torso posture detected.")
                logger.info("Torso posture looks good")

            # -------------------------
            # Knee lift
            # -------------------------
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

            if knee.y < hip.y:
                feedback.append(
                    "Good knee lift detected, indicating efficient stride mechanics."
                )
                logger.info("Good knee lift detected")
            else:
                feedback.append(
                    "Low knee lift detected. Focus on driving knees slightly higher."
                )
                logger.info("Low knee lift detected")

            # -------------------------
            # Foot strike estimation
            # -------------------------
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]

            if heel.y < ankle.y:
                feedback.append(
                    "Possible heel strike detected. Consider a mid-foot landing for better efficiency."
                )
                logger.info("Heel strike pattern detected")

            logger.info("Posture analysis completed successfully")
            return feedback

    except Exception as e:
        logger.exception("Posture analysis failed due to an exception")
        return ["Internal error during posture analysis. Please try another image."]
