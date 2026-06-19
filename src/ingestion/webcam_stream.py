import cv2

def get_webcam_stream(camera_index: int = 0):
    """
    Opens the webcam and yields frames one at a time.
    camera_index=0 is usually the default/built-in webcam.
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

