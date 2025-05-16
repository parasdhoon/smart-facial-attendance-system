import streamlit as st
import cv2
import pickle
import numpy as np
from datetime import datetime
from utils.helper_functions import get_face_embedding, cosine_similarity
from yolov5 import YOLOv5

EMBEDDINGS_FILE = "database/embeddings.pkl"
ATTENDANCE_FILE = "database/attendance.csv"

model = YOLOv5("yolov5s.pt") 

def mark_attendance():
    st.title("📸 Mark Your Attendance")
    st.info("Press 's' to scan your face or 'q' to quit.")
    st.write("Initializing webcam...")

    cap = cv2.VideoCapture(0)

    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        st.error("No registered faces found. Please register first.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame)
        faces = results.xyxy[0] 

        for (x1, y1, x2, y2, conf, cls) in faces:
            if cls == 0: 
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.imshow("Mark Attendance - Press 's' to Scan", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.destroyAllWindows()
            cap.release()

            recognized_any = False
            for (x1, y1, x2, y2, conf, cls) in faces:
                if cls == 0:
                    face = frame[int(y1):int(y2), int(x1):int(x2)] 

                    embedding = get_face_embedding(face)
                    if embedding is None:
                        continue  
                    
                    recognized = False
                    for entry in data:
                        sim = cosine_similarity(embedding, entry['embedding'])
                        if sim > 0.7:
                            name = entry['name']
                            user_id = entry['user_id']
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            with open(ATTENDANCE_FILE, "a") as log:
                                log.write(f"{name},{user_id},{timestamp}\n")

                            st.success(f"✅ Attendance marked for **{name}** at {timestamp}")
                            recognized = True
                            break
                    
                    if not recognized:
                        st.warning("🙁 Face not recognized. Please register first.")
                    else:
                        recognized_any = True

            if not recognized_any:
                st.warning("🙁 No recognized faces found. Please try again.")
            break

        elif key == ord('q'):
            st.info("Exited attendance marking.")
            break

    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()
