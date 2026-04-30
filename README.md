# ICU Sepsis Early Warning System

Final Year Project – Maryam Chaudhry (w1916669)

This project is an ICU sepsis early-warning prototype and uses machine learning to estimate whether a patient may be at risk of developing sepsis within the next 6 hours, then displays the prediction through a Streamlit dashboard.

The aim is not to replace clinical judgement or diagnose sepsis by itself. The system is a prototype designed to demonstrate early-warning prediction, explainability, threshold control and audit logging.

---

## Main Features

- Machine learning pipeline for six-hour sepsis early-warning prediction
- A final saved model bundle using Random Forest, with Logistic Regression used as a baseline
- Patient-level train, validation and test splitting to reduce data leakage
- Amber and red alert thresholds
- Alert hygiene rules, including consecutive-hit logic and cooldown
- SHAP-based local explanations for flagged patients
- Streamlit dashboard with three user roles:
  - Clinician
  - Admin
  - Manager
- Patient search, risk score display, risk trend and latest vitals
- Threshold adjustment page for admins
- Manager performance page with confusion matrix views
- CSV-based prototype audit log for logins, acknowledgements and threshold changes

---

## Repository Structure

```text

data/

    Demo data, saved test evaluation files, model results and threshold/audit files.

models/

    Saved trained model bundle used by the Streamlit app.

notebooks/

    Training and experimentation notebook.

src/

    Supporting model/service utilities.

streamlit_app/

    Streamlit dashboard application.

requirements.txt

    Packages needed to run the Streamlit dashboard.

requirements-train.txt

    Packages needed to run the training notebook.

README.md

    Project instructions and overview.

SCOPE.md

    Project scope and boundary notes.

    ## Troubleshooting

If VS Code shows yellow squiggly lines under imports such as `streamlit`, `shap`, `pandas` or `sklearn`, it usually means VS Code is using the wrong Python interpreter or the dependencies have not been installed in the active environment.

To fix this, select the project virtual environment in VS Code:

```text
Python: Select Interpreter


## Also check this before pushing

Your `.venv` folder is visible in VS Code. That is fine locally, but do **not** upload it to GitHub. Your `.gitignore` should include:

```gitignore
.venv/
venv/
__pycache__/
.ipynb_checkpoints/
.DS_Store