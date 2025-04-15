import os
import pickle
import cv2
import numpy as np
import pandas as pd
from keras_facenet import FaceNet
import streamlit as st

embedder = FaceNet()


def get_face_embedding(frame):
    """
    Extract the facial embedding of a given frame using the FaceNet model.
    """
    faces = embedder.extract(frame, threshold=0.95)
    if len(faces) == 0:
        return None
    return faces[0]['embedding']


def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two vectors (embeddings).
    """
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def save_user_data(name, user_id, embedding, file_path):
    """
    Save the user's data, including the name, user_id, and face embedding.
    """
    data = []
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

    data.append({"name": name, "user_id": user_id, "embedding": embedding})

    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def view_attendance():
    """
    View the attendance data from a CSV file.
    """
    file_path = "database/attendance.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, names=["Name", "User ID", "Timestamp"])
        st.dataframe(df[::-1])  # Show latest on top
    else:
        st.info("No attendance logs found yet.")


def is_live_face(frame):
    """
    A simple liveness detection function using face landmarks and eye-blink detection.
    In this example, we use OpenCV to detect eye-blinks.
    More sophisticated models can be used for real-time liveness detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Simple liveness detection: If no eyes are detected, consider the face not live
        if len(eyes) == 0:
            return False
        
        # Check if there's more than one eye detected (this is a very basic heuristic)
        if len(eyes) >= 2:
            return True
    
    return False
