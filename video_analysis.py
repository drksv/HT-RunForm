import cv2
import mediapipe as mp
import tempfile

mp_pose = mp.solutions.pose

def analyze_running_video(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        video_path = f.name

    cap = cv2.VideoCapture(video_path)
    feedback = []

    with mp_pose.Pose() as pose:
        frame_count = 0
        while cap.isOpened() and frame_count < 60:
            ret, frame = cap.read()
            if not ret:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
                toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

                if heel.y < toe.y:
                    feedback.append("Possible heel striking detected.")

            frame_count += 1

    cap.release()
    return list(set(feedback))
