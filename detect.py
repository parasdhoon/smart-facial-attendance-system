# detect.py
import streamlit as st
import cv2
import pickle
import numpy as np
from datetime import datetime
from utils.helper_functions import get_face_embedding, cosine_similarity, is_live_face

EMBEDDINGS_FILE = "database/embeddings.pkl"
ATTENDANCE_FILE = "database/attendance.csv"


def mark_attendance():
    st.write("Starting webcam for attendance...")

    cap = cv2.VideoCapture(0)
    recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Mark Attendance - Press 's' to Scan", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            # Check for liveness
            if not is_live_face(frame):
                st.error("Liveness test failed. Please try again with a real face.")
                break

            embedding = get_face_embedding(frame)

            if embedding is None:
                st.error("Could not detect face. Try again.")
                break

            with open(EMBEDDINGS_FILE, "rb") as f:
                data = pickle.load(f)

            for entry in data:
                sim = cosine_similarity(embedding, entry['embedding'])
                if sim > 0.7:
                    name = entry['name']
                    user_id = entry['user_id']
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(ATTENDANCE_FILE, "a") as log:
                        log.write(f"{name},{user_id},{timestamp}\n")
                    st.success(f"Attendance marked for {name} at {timestamp}")
                    recognized = True
                    break

            if not recognized:
                st.warning("Face not recognized.")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
