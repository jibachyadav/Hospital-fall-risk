import sys
sys.path.append(".")

import cv2
import mediapipe as mp
from src.ingestion.webcam_stream import get_webcam_stream
from src.ingestion.pose_extractor import extract_pose_landmarks

mp_pose = mp.solutions.pose

print("Testing pose extraction. Press 'q' to quit.")

with mp_pose.Pose() as pose_model:
    for frame in get_webcam_stream():
        landmarks = extract_pose_landmarks(frame, pose_model)

        if landmarks is not None:
            print(f"Detected {len(landmarks)} landmarks, shape: {landmarks.shape}")
        else:
            print("No person detected")

        cv2.imshow("Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print("Done.")

