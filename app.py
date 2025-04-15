# app.py
import streamlit as st
from register import register_user
from detect import mark_attendance
from utils.helper_functions import view_attendance

st.set_page_config(page_title="Facial Attendance System", layout="centered")
st.title("\U0001F4BC Smart Facial Attendance System")

option = st.sidebar.selectbox("Choose Option", ["Register", "Mark Attendance", "View Attendance Logs"])

if option == "Register":
    st.header("Register a New User")
    register_user()

elif option == "Mark Attendance":
    st.header("Mark Attendance")
    mark_attendance()

elif option == "View Attendance Logs":
    st.header("Attendance Logs")
    view_attendance()