import sys
sys.path.append(".")

import cv2
from src.ingestion.webcam_stream import get_webcam_stream

print("Starting webcam stream. Press 'q' to quit.")

for frame in get_webcam_stream():
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Stream stopped.")

