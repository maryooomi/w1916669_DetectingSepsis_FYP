# ICU Sepsis Early Warning System

Final Year Project – Maryam Chaudhry (w1916669)

This project builds a machine learning system to **predict sepsis early in ICU patients** using vital signs and laboratory data.

The system includes:

- A trained machine learning model (Random Forest / Logistic Regression)
- Early warning risk predictions
- SHAP explanations showing why a patient is flagged
- A Streamlit dashboard for clinicians, managers, and admins

---

# Run the Streamlit App

Install dependencies:

pip install -r requirements.txt


Run the dashboard:


streamlit run streamlit_app/app.py


---

# Demo Accounts

**Clinician**

email: clinician@icu.local  
password: clinician123  

**Admin**

email: admin@icu.local  
password: admin123  

**Manager**

email: manager@icu.local  
password: manager123  

---

# Repository Structure


data/ demo dataset used by the dashboard
models/ trained ML model bundle
notebooks/ experimentation notebooks
src/ model and preprocessing utilities
streamlit_app/ Streamlit dashboard code


---

# Project Goal

The goal is **not to diagnose sepsis after it happens**, but to **predict the risk that a patient may develop sepsis within the next 6 hours**, giving clinicians time to intervene earlier.
