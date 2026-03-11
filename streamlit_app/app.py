import streamlit as st

# Page configuration
st.set_page_config(page_title="ICU Sepsis Early Warning System", layout="wide")

st.title("ICU Sepsis Early Warning System")

st.write("Login to access the dashboard.")

# Fake user database
users = {
    "clinician@icu.local": {"password": "clinician123", "role": "Clinician"},
    "admin@icu.local": {"password": "admin123", "role": "Admin"},
    "manager@icu.local": {"password": "manager123", "role": "Manager"},
}

# Login form
email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):

    if email in users and users[email]["password"] == password:

        role = users[email]["role"]

        st.success(f"Logged in as {role}")

        if role == "Clinician":
            st.write("Clinician dashboard coming next")

        elif role == "Admin":
            st.write("Admin settings page coming next")

        elif role == "Manager":
            st.write("Manager performance dashboard coming next")

    else:
        st.error("Invalid login details")
