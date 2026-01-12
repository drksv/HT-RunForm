import cv2
import tempfile
import os
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
# Video analysis
# --------------------------------------------------
def analyze_running_video(video_bytes: bytes):
    feedback = set()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        video_path = f.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return ["Invalid video uploaded"]

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            frame_count = 0
            while cap.isOpened() and frame_count < 60:
                ret, frame = cap.read()
                if not ret:
                    break

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    heel = lm[mp_pose.PoseLandmark.LEFT_HEEL]
                    toe = lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

                    if heel.y < toe.y:
                        feedback.add("Possible heel striking detected.")

                frame_count += 1

        if not feedback:
            feedback.add("Running form appears consistent.")

        logger.info("Running video analysis completed")
        return list(feedback)

    except Exception as e:
        logger.exception("Running video analysis failed")
        return [f"Video analysis failed: {str(e)}"]

    finally:
        cap.release()
        os.remove(video_path)
