# register.py
import streamlit as st
import cv2
import os
import numpy as np
import pickle
from datetime import datetime
from utils.helper_functions import get_face_embedding, save_user_data

DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "database/embeddings.pkl"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

if not os.path.exists("database"):
    os.makedirs("database")


def register_user():
    name = st.text_input("Enter your name:")
    user_id = st.text_input("Enter a unique user ID:")

    if st.button("Capture Face"):
        if not name or not user_id:
            st.warning("Please fill in both fields.")
            return

        cap = cv2.VideoCapture(0)
        st.info("Press 's' to capture your face.")
        captured = False

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow("Capture Face", frame)
            key = cv2.waitKey(1)

            if key == ord('s'):
                captured = True
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured:
            user_folder = os.path.join(DATASET_DIR, user_id)
            os.makedirs(user_folder, exist_ok=True)

            img_path = os.path.join(user_folder, f"{user_id}.jpg")
            cv2.imwrite(img_path, frame)

            embedding = get_face_embedding(frame)

            if embedding is not None:
                save_user_data(name, user_id, embedding, EMBEDDINGS_FILE)
                st.success(f"User {name} registered successfully!")
            else:
                st.error("Failed to extract face. Try again.")
