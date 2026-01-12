import cv2
import numpy as np
import tempfile
import os
import logging

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import Image, ImageFormat

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model setup (reuse same model)
# --------------------------------------------------
MODEL_PATH = "pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Pose model not found. Ensure posture.py downloads it first.")

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(options)
logger.info("MediaPipe Pose Landmarker (video) initialized")

# --------------------------------------------------
# Video analysis
# --------------------------------------------------
def analyze_running_video(video_bytes: bytes):
    """
    Analyze a running video and return posture / gait feedback.
    """

    feedback = set()

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        video_path = f.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Unable to open uploaded video")
        return ["Invalid video uploaded"]

    frame_count = 0
    MAX_FRAMES = 60

    try:
        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = Image(
                image_format=ImageFormat.SRGB,
                data=rgb_frame
            )

            result = pose_landmarker.detect(mp_image)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                heel = landmarks[29]  # LEFT_HEEL
                toe = landmarks[31]   # LEFT_FOOT_INDEX

                if heel.y < toe.y:
                    feedback.add(
                        "Possible heel striking detected. Consider transitioning toward a mid-foot strike."
                    )

            frame_count += 1

        if not feedback:
            feedback.add("Running form appears generally consistent in sampled frames.")

        logger.info("Running video analysis completed")
        return list(feedback)

    except Exception as e:
        logger.exception("Running video analysis failed")
        return [f"Video analysis failed: {str(e)}"]

    finally:
        cap.release()
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp video file: {e}")







