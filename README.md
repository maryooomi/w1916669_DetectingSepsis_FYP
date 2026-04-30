# ICU Sepsis Early Warning System

Final Year Project – Maryam Chaudhry (w1916669)

This project is an ICU sepsis early-warning prototype. It uses machine learning to estimate whether a patient may be at risk of developing sepsis within the next 6 hours, then presents the result through a Streamlit dashboard.

The system is not intended to replace clinical judgement or diagnose sepsis on its own. It is a prototype built to demonstrate early-warning prediction, explainability, threshold control and audit logging in a clear and testable way.

---

## Main Features

- Machine learning pipeline for six-hour sepsis early-warning prediction
- Final saved Random Forest model, with Logistic Regression used as a baseline
- Patient-level train, validation and test splitting to reduce data leakage
- Amber and red alert thresholds
- Alert hygiene rules, including consecutive-hit logic and cooldown
- SHAP-based local explanations for flagged patients
- Streamlit dashboard with three user roles:
  - Clinician
  - Admin
  - Manager
- Patient search, current risk score, latest vitals and risk trend
- Admin threshold adjustment page
- Manager performance page with confusion matrix views
- CSV-based prototype audit log for logins, alert acknowledgements and threshold changes

---

## Repository Structure

```text
data/
    Demo data, saved test evaluation files, model results and dashboard data.

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
    Project overview and run instructions.

SCOPE.md
    Project scope and boundary notes.
```

---

## Required Files

The Streamlit dashboard expects these files to be present:

```text
models/sepsis_model_bundle.pkl
data/demo_df.parquet
data/demo_X.parquet
data/demo_metadata.json
data/model_results.csv
data/test_eval_with_alerts.parquet
```

The app can create `audit_log.csv` automatically if it does not already exist.

The `active_thresholds.json` file is optional. It is only used after an admin has saved threshold changes.

---

## How to Run the Streamlit App

Run these commands from the project root folder.

### Step 1: Open the project folder in VS Code

Open the folder that contains:

```text
README.md
requirements.txt
streamlit_app/
data/
models/
```

### Step 2: Create and activate a virtual environment

For macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install the dashboard dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
python -m streamlit run streamlit_app/app.py
```

Streamlit should open a local browser page. If it does not open automatically, copy the local URL shown in the terminal.

---

## Demo Login Accounts

### Clinician

```text
Email: clinician@icu.local
Password: clinician123
```

### Admin

```text
Email: admin@icu.local
Password: admin123
```

### Manager

```text
Email: manager@icu.local
Password: manager123
```

---

## How to Run the Training Notebook

The final training notebook is stored in:

```text
notebooks/training-2.ipynb
```

To install the training dependencies:

```bash
python -m pip install -r requirements-train.txt
```

The notebook downloads the dataset, creates the six-hour early-warning target, trains candidate models, selects the final model using validation results, evaluates it on the held-out test set and exports the model/data artefacts used by the dashboard.

---

## Troubleshooting

If VS Code shows yellow squiggly lines under imports such as `streamlit`, `shap`, `pandas` or `sklearn`, it usually means VS Code is using the wrong Python interpreter or the dependencies have not been installed in the active environment.

To fix this, select the project virtual environment in VS Code:

```text
Python: Select Interpreter
```

Then choose the interpreter inside `.venv`.

After that, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

If the dashboard opens in the browser, the warning is only an editor/environment warning and not a code error.

---

## Important Prototype Limitations

This project is a working prototype, not a live clinical system. It does not connect to a real Electronic Health Record system, hospital database or external clinical API.

The login system uses hard-coded demonstration accounts. The audit log is CSV-based. These choices are suitable for demonstrating the project, but they are not suitable for production use.

Before any real clinical deployment, the system would require external validation, stronger authentication, secure database storage, clinical safety review, regulatory assessment and integration with approved hospital infrastructure.

---

## Project Goal

The goal of this project is to demonstrate how machine learning can support earlier awareness of sepsis risk in ICU patients. The dashboard makes the model output more usable by showing risk scores, alert states, explanations, threshold controls and audit information.
