import cv2
import numpy as np
import torch
from deepface import DeepFace
import time
import os
from yolov5 import YOLOv5  # Import YOLOv5 class

# Load YOLOv5 model (face detection model or the generic YOLOv5 model)
yolo_model = YOLOv5("yolov5s.pt")  # Use yolov5s.pt or yolov5-face.pt if you have a dedicated face model

def preprocess_frame(frame):
    """
    Preprocess the frame to match the input format of DeepFace for liveness detection.
    """
    face = cv2.resize(frame, (100, 100))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def detect_faces_yolo(frame):
    """
    Detect faces using YOLOv5.
    """
    # Perform inference with YOLOv5 to detect faces
    results = yolo_model.predict(frame)  # Perform inference
    faces = []

    # Results contains predictions in the format [xywh, confidence, class]
    for *xywh, conf, cls in results.xywh[0]:
        if int(cls) == 0 and conf > 0.5:  # Class 0 typically corresponds to 'person'
            x1, y1, w, h = map(int, xywh)
            faces.append((x1, y1, w, h))  # Store face bounding boxes

    return faces

def record_and_check_liveness_and_get_last_frame(duration=3, threshold=0.7):
    """
    Record a short video and check liveness using DeepFace.
    """
    video_path = "temp/liveness_check.avi"
    cap = cv2.VideoCapture(0)
    fps = 20.0
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frames = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using YOLOv5
        faces = detect_faces_yolo(frame)

        if faces:
            # Draw bounding boxes around detected faces (optional)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture the frame for liveness detection
            frames.append(frame.copy())
            out.write(frame)
        
        cv2.imshow("Recording for Liveness", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not frames:
        return False, None

    # Liveness detection using DeepFace
    live_count = 0
    total = 0
    for frame in frames:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if 'dominant_emotion' in result:
                # Example: Check if the dominant emotion or other metric suggests the person is alive
                live_count += 1
            total += 1
        except Exception as e:
            print(f"Error in liveness detection: {e}")
    
    is_live = (live_count / total) >= threshold
    last_frame = frames[-1] if is_live else None

    os.remove(video_path)
    return is_live, last_frame
