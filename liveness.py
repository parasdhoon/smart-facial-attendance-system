# liveness.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/liveness_model.h5")


def is_live_face(frame):
    try:
        face = cv2.resize(frame, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0]
        return prediction[1] > 0.9  # Class 1 means live face

    except Exception as e:
        print("Liveness detection error:", e)
        return False