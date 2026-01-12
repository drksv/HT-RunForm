import cv2
import mediapipe as mp
import numpy as np


# ✅ Initialize MediaPipe Pose correctly
mp_pose = mp.solutions.pose

def analyze_posture_image(image_bytes):
    # Convert bytes to OpenCV image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return ["Invalid image uploaded"]

    # Run MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return ["No runner detected. Please upload a full-body running image."]

        landmarks = results.pose_landmarks.landmark
        feedback = []

        # --- Posture checks ---
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        if shoulder.y > hip.y:
            feedback.append(
                "You appear to be leaning forward excessively. Try maintaining a tall, upright running posture."
            )
        else:
            feedback.append("Good upright torso posture detected.")

        # --- Knee lift check ---
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]

        if knee.y < hip.y:
            feedback.append("Good knee lift detected, indicating efficient stride mechanics.")
        else:
            feedback.append("Low knee lift detected. Focus on driving knees slightly higher.")

        # --- Foot strike estimation ---
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]

        if heel.y < ankle.y:
            feedback.append("Possible heel strike detected. Consider a mid-foot landing for efficiency.")

        return feedback
