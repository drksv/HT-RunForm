import cv2
import numpy as np
import logging
import mediapipe as mp

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose

# --------------------------------------------------
# Image analysis
# --------------------------------------------------
def analyze_posture_image(image_bytes: bytes):
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning("Invalid image uploaded")
            return ["Invalid image uploaded"]

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5
        ) as pose:

            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                return ["No runner detected. Please upload a full-body running image."]

            lm = results.pose_landmarks.landmark
            feedback = []

            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]

            if shoulder.y > hip.y:
                feedback.append("You appear to be leaning forward excessively.")
            else:
                feedback.append("Good upright torso posture detected.")

            if knee.y < hip.y:
                feedback.append("Good knee lift detected.")
            else:
                feedback.append("Low knee lift detected.")

            if heel.y < ankle.y:
                feedback.append("Possible heel strike detected.")

            logger.info("Posture image analysis completed")
            return feedback

    except Exception as e:
        logger.exception("Posture analysis failed")
        return [f"Posture analysis failed: {str(e)}"]
