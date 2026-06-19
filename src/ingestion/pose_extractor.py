import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_pose_landmarks(frame, pose_model):
    """
    Takes a single BGR frame (from OpenCV) and a mediapipe Pose model.
    Returns a numpy array of 33 (x, y, z, visibility) landmarks,
    or None if no person is detected.
    """
    rgb_frame = frame[:, :, ::-1]  # mediapipe expects RGB, OpenCV gives BGR
    results = pose_model.process(rgb_frame)

    if not results.pose_landmarks:
        return None

    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(landmarks)
