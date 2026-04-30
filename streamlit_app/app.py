# ICU Sepsis Early Warning System - Streamlit Prototype

# This app loads the trained sepsis model and saved test/demo data,
# then displays role-based dashboards for clinicians, admins and managers.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Project folders and saved files

# The model training is done separately in the notebook.
# This Streamlit app only loads the saved model, data and settings.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
THRESHOLDS_PATH = DATA_DIR / "active_thresholds.json"
AUDIT_LOG_PATH = DATA_DIR / "audit_log.csv"


DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="ICU Sepsis Early Warning System",
    layout="wide"
)



# Prototype user accounts

users = {
    "clinician@icu.local": {"password": "clinician123", "role": "Clinician"},
    "admin@icu.local": {"password": "admin123", "role": "Admin"},
    "manager@icu.local": {"password": "manager123", "role": "Manager"},
}


# Session state

# Keep the login details when Streamlit reruns the page.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None


# Reset the saved login details and reload the app.
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user_email = None
    st.rerun()


#Login page

# Shows a simple login screen and assigns the user their role.
def login_page():

    st.title("ICU Sepsis Early Warning System")
    st.write("Login to access the dashboard.")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if email in users and users[email]["password"] == password:

            st.session_state.logged_in = True
            st.session_state.role = users[email]["role"]
            st.session_state.user_email = email

            # Record the successful login so the audit log shows user activity.
            append_audit_event(
                action="login_success",
                details={"login_role": users[email]["role"]}
            )

            st.rerun()

        else:
            st.error("Invalid login details")


# General helper functions


# Use the ICU time column where it is available.
def get_time_col(df: pd.DataFrame) -> str:
    return "ICULOS" if "ICULOS" in df.columns else "Hour"

# Make model feature names easier to read on the dashboard.
def pretty_feature_name(name: str) -> str:
    name = str(name).replace("num__", "")
    if "missingindicator_" in name:
        base = name.replace("missingindicator_", "")
        return f"{base} missing"
    return name


# Explains one patient prediction by showing which features had the biggest effect.
# This makes the model output more transparent than just showing a risk score.
def compute_local_explanation_df(bundle: dict, X_one_row: pd.DataFrame) -> pd.DataFrame:

    pipe = bundle["base_model"]
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # Use the same feature columns and preprocessing that were used during training.
    X_input = X_one_row[bundle["feature_columns"]].copy()
    X_tx = preprocess.transform(X_input)
    X_tx = X_tx.toarray() if hasattr(X_tx, "toarray") else np.asarray(X_tx)

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_tx.shape[1])]

    feature_names = [pretty_feature_name(f) for f in feature_names]
    model_name = model.__class__.__name__


    # Random Forest predictions are explained using Tree SHAP.
    if model_name == "RandomForestClassifier":
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_tx)

        if isinstance(sv, list):
            contrib = np.asarray(sv[1][0])
        else:
            sv = np.asarray(sv)

            if sv.ndim == 3 and sv.shape[-1] == 2:
                contrib = sv[0, :, 1]
            elif sv.ndim == 2:
                contrib = sv[0]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")

        method = "Tree SHAP"


    # Logistic Regression is explained using coefficient contribution.
    elif model_name == "LogisticRegression":
        coefs = np.asarray(model.coef_).ravel()
        row = np.asarray(X_tx[0]).ravel()

        if len(coefs) != len(row):
            raise ValueError(
                f"Coefficient length ({len(coefs)}) does not match "
                f"transformed row length ({len(row)})"
            )

        contrib = row * coefs
        method = "Logistic contribution"

    else:
        raise ValueError(f"Unsupported model type for local explanation: {model_name}")


    # Store the explanation in a table so it can be displayed clearly in Streamlit.
    out = pd.DataFrame({
        "feature": feature_names,
        "contribution": contrib
    })

    out["abs_contribution"] = out["contribution"].abs()
    out["effect"] = np.where(
        out["contribution"] >= 0,
        "pushes risk up",
        "pushes risk down"
    )
    out["method"] = method
    out["model_type"] = model_name

    return out.sort_values("abs_contribution", ascending=False).reset_index(drop=True)

GROUP_COL = "Patient_ID"


# Loading saved model, data and evaluation files


# Load the trained model bundle created in the notebook.
def load_bundle():
    return joblib.load(MODELS_DIR / "sepsis_model_bundle.pkl")

# Load thresholds saved by the admin page, if they exist.
def load_persisted_thresholds():
    if not THRESHOLDS_PATH.exists():
        return None

    try:
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

        amber = float(payload["amber"])
        red = float(payload["red"])


        # Keep thresholds in the valid probability range.
        amber = max(0.0, min(1.0, amber))
        red = max(0.0, min(1.0, red))

        # Amber should be lower than or equal to red.
        if amber > red:
            amber = red

        return {
            "amber": amber,
            "red": red,
            "saved_at": payload.get("saved_at")
        }
    except Exception:

        # If the file is missing or corrupted, fall back to model defaults.
        return None



# Save changed thresholds without changing the trained model.
def save_persisted_thresholds(amber_thr, red_thr):
    amber_thr = float(max(0.0, min(1.0, amber_thr)))
    red_thr = float(max(0.0, min(1.0, red_thr)))

    if amber_thr > red_thr:
        amber_thr = red_thr

    payload = {
        "amber": amber_thr,
        "red": red_thr,
        "saved_at": pd.Timestamp.utcnow().isoformat()
    }

    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Remove saved admin thresholds and return to the model defaults.
def clear_persisted_thresholds():
    if THRESHOLDS_PATH.exists():
        THRESHOLDS_PATH.unlink()


# Load demo patients saved from the held-out test split.
def load_demo_assets():
    df = pd.read_parquet(DATA_DIR / "demo_df.parquet")
    demo_X = pd.read_parquet(DATA_DIR / "demo_X.parquet")
    return df, demo_X


# Load metadata showing where the demo patients came from.
def load_demo_metadata():
    meta_path = DATA_DIR / "demo_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Load the held-out test evaluation data used by the manager dashboard.
def load_eval_frame():
    eval_path_parquet = DATA_DIR / "test_eval_with_alerts.parquet"
    eval_path_csv = DATA_DIR / "test_eval_with_alerts.csv"

    if eval_path_parquet.exists():
        return pd.read_parquet(eval_path_parquet)
    return pd.read_csv(eval_path_csv)



# Audit log functions


# The audit log records important user actions such as logins,
# threshold changes and alert acknowledgements.
AUDIT_COLUMNS = [
    "timestamp_utc",
    "user_email",
    "role",
    "action",
    "patient_id",
    "observation_time",
    "alert_state",
    "risk_score",
    "details",
]

# Create an empty audit file if it has not been created yet.
def ensure_audit_log():
    if not AUDIT_LOG_PATH.exists():
        pd.DataFrame(columns=AUDIT_COLUMNS).to_csv(AUDIT_LOG_PATH, index=False)


# Convert IDs and times into consistent strings so records match correctly.
def normalise_key(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass

    return str(value)


# Convert numeric alert states into readable dashboard labels.
def alert_state_to_label(value):
    try:
        v = int(float(value))
    except Exception:
        return ""

    return {
        0: "🟢 OK",
        1: "🟠 WATCH",
        2: "🔴 ALERT",
    }.get(v, str(v))



# Read extra audit details that were saved as JSON text.
def parse_audit_details(raw):
    if raw is None:
        return {}
    try:
        if pd.isna(raw):
            return {}
    except Exception:
        pass

    raw = str(raw).strip()
    if raw == "":
        return {}

    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


# Store extra audit details as JSON.
def append_audit_event(
    action: str,
    patient_id=None,
    observation_time=None,
    alert_state=None,
    risk_score=None,
    details=None
):
    

    # Add one new row to the audit log.
    # This gives evidence of what happened in the prototype and who did it.
    ensure_audit_log()

    row = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "user_email": st.session_state.get("user_email", "unknown"),
        "role": st.session_state.get("role", "Unknown"),
        "action": action,
        "patient_id": normalise_key(patient_id),
        "observation_time": normalise_key(observation_time),
        "alert_state": "" if alert_state is None else int(alert_state),
        "risk_score": "" if risk_score is None else float(risk_score),
        "details": json.dumps(details or {}, ensure_ascii=False, default=str),
    }

    pd.DataFrame([row], columns=AUDIT_COLUMNS).to_csv(
        AUDIT_LOG_PATH,
        mode="a",
        header=False,
        index=False
    )


# Load the audit log and convert timestamps into readable dates.
def load_audit_log():
    ensure_audit_log()
    df = pd.read_csv(AUDIT_LOG_PATH)

    if not df.empty and "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            errors="coerce",
            utc=True
        )

    return df


# Check whether the selected alert has already been acknowledged.
def get_acknowledgement_for_alert(patient_id, observation_time):
    audit_df = load_audit_log()
    if audit_df.empty:
        return None

    patient_key = normalise_key(patient_id)
    obs_key = normalise_key(observation_time)

    ack_df = audit_df[
        (audit_df["action"].astype(str) == "alert_acknowledged") &
        (audit_df["patient_id"].astype(str) == patient_key) &
        (audit_df["observation_time"].astype(str) == obs_key)
    ].sort_values("timestamp_utc", ascending=False)

    if ack_df.empty:
        return None

    return ack_df.iloc[0].to_dict()




# Alert rules and threshold handling


# Read the alert hygiene rules stored in the model bundle.
# These rules reduce noisy alerts by using persistence and cooldown behaviour.

def get_alert_policy(bundle):
    policy = bundle.get("threshold_policy", {})
    return {
        "k_amber": int(policy.get("k_consecutive_amber", 2)),
        "k_red": int(policy.get("k_consecutive_red", 1)),
        "cooldown_hours": int(policy.get("cooldown_hours", 6)),
        "allow_escalation": bool(policy.get("allow_escalation_to_red", True)),
    }



# Start the app with saved admin thresholds if available.
# Otherwise, use the default thresholds saved with the model.
def init_threshold_state(bundle):
    persisted = load_persisted_thresholds()

    if persisted is not None:
        start_amber = float(persisted["amber"])
        start_red = float(persisted["red"])
    else:
        start_amber = float(bundle["thresholds"]["amber"])
        start_red = float(bundle["thresholds"]["red"])

    if "amber_threshold" not in st.session_state:
        st.session_state.amber_threshold = start_amber

    if "red_threshold" not in st.session_state:
        st.session_state.red_threshold = start_red


# Return the thresholds currently being used in the session.
def get_active_thresholds():
    amber_thr = float(st.session_state.amber_threshold)
    red_thr = float(st.session_state.red_threshold)

    if amber_thr > red_thr:
        amber_thr = red_thr
        st.session_state.amber_threshold = amber_thr

    return amber_thr, red_thr


# Reset the session thresholds back to the model defaults.
def reset_thresholds_to_defaults(bundle):
    st.session_state.amber_threshold = float(bundle["thresholds"]["amber"])
    st.session_state.red_threshold = float(bundle["thresholds"]["red"])


# Convert model probabilities into three simple alert levels.
# 0 = OK, 1 = WATCH, 2 = ALERT.
def score_to_level(scores, thr_amber, thr_red):
    levels = np.zeros(len(scores), dtype=np.int8)
    levels[scores >= thr_amber] = 1
    levels[scores >= thr_red] = 2
    return levels


def apply_alert_hygiene(
    eval_df,
    patient_col,
    time_col,
    score_col,
    thr_amber,
    thr_red,
    k_amber=2,
    k_red=1,
    cooldown_hours=6,
    allow_escalation=True
):
    

    # Apply the alert policy patient by patient.
    # This prevents the dashboard from firing a new alert every time the model score is high.
    out = eval_df.copy()
    out["level_raw"] = score_to_level(out[score_col].values, thr_amber, thr_red)
    out["alert_state"] = 0
    out["alert_fired"] = 0


    # Keep each patient's observations in time order.
    for pid, g in out.groupby(patient_col, sort=False):
        g = g.sort_values(time_col)
        idx = g.index

        consec_any = 0
        consec_red = 0
        state = 0
        state_start_time = -1e18

        for i in idx:
            t = float(out.at[i, time_col])
            level = int(out.at[i, "level_raw"])


            # Expire an active alert after the cooldown period.
            if state != 0 and (t - state_start_time) >= cooldown_hours:
                state = 0


            # Count how many amber-or-higher scores happen in a row.
            if level >= 1:
                consec_any += 1
            else:
                consec_any = 0

            # Count how many red scores happen in a row.
            if level == 2:
                consec_red += 1
            else:
                consec_red = 0

            fired = 0

            # Create a new WATCH or ALERT state when the rules are met.
            if state == 0:
                if consec_red >= k_red:
                    state = 2
                    state_start_time = t
                    fired = 2
                elif consec_any >= k_amber:
                    state = 1
                    state_start_time = t
                    fired = 1
                      
                      
            # Escalate WATCH to ALERT if the patient becomes high risk.
            elif state == 1 and allow_escalation and (consec_red >= k_red): 
                state = 2
                state_start_time = t
                fired = 2

            out.at[i, "alert_state"] = state
            out.at[i, "alert_fired"] = fired

    return out




# Initialise thresholds before showing any page
bundle_for_state = load_bundle()
init_threshold_state(bundle_for_state)




# Clinician dashboard


# This page shows monitored patients, risk scores, trends,
# local explanations and alert acknowledgement.
def clinician_dashboard():

    st.title("ICU Sepsis Early Warning System")
    st.header("Clinician Dashboard")

    bundle = load_bundle()
    loaded_model_name = bundle["base_model"].named_steps["model"].__class__.__name__
    st.caption(f"Loaded final model: {loaded_model_name}")

    df, demo_X = load_demo_assets()

    demo_meta = load_demo_metadata()
    if demo_meta and demo_meta.get("all_from_test_split"):
        st.caption("Demo data source: held-out test patients only.")

    df = df.copy()
    demo_X = demo_X.copy()

    amber_thr, red_thr = get_active_thresholds()
    policy = get_alert_policy(bundle)
    time_col = get_time_col(df)

    st.caption(
        f"Active alert policy in this session: "
        f"amber={amber_thr:.4f}, red={red_thr:.4f}, "
        f"k_amber={policy['k_amber']}, k_red={policy['k_red']}, "
        f"cooldown={policy['cooldown_hours']}h"
    )


    # Generate a risk score for every demo observation.
    df["risk_score"] = bundle["base_model"].predict_proba(
        demo_X[bundle["feature_columns"]]
    )[:, 1]


    # Convert risk scores into OK, WATCH or ALERT using the alert rules.
    alert_df = apply_alert_hygiene(
        df[[GROUP_COL, time_col, "risk_score"]].copy(),
        patient_col=GROUP_COL,
        time_col=time_col,
        score_col="risk_score",
        thr_amber=amber_thr,
        thr_red=red_thr,
        k_amber=policy["k_amber"],
        k_red=policy["k_red"],
        cooldown_hours=policy["cooldown_hours"],
        allow_escalation=policy["allow_escalation"],
    )

    df["alert_state"] = alert_df["alert_state"].values
    df["alert_fired"] = alert_df["alert_fired"].values

    df["alert"] = np.select(
        [df["alert_state"] == 2, df["alert_state"] == 1],
        ["🔴 ALERT", "🟠 WATCH"],
        default="🟢 OK"
    )


    # Show one latest row per patient on the main dashboard.
    latest = (
        df.sort_values(time_col)
        .groupby(GROUP_COL)
        .tail(1)
        .copy()
    )

    col1, col2, col3 = st.columns(3)

    high_alerts = int((latest["alert_state"] == 2).sum())
    watch_alerts = int((latest["alert_state"] == 1).sum())

    col1.metric("Patients monitored", len(latest))
    col2.metric("High risk alerts", high_alerts)
    col3.metric("Watch alerts", watch_alerts)



    # Allow the clinician to quickly find a patient.
    search = st.text_input("Search patient ID")

    filtered_latest = latest.copy()
    if search:
        filtered_latest = filtered_latest[
            filtered_latest[GROUP_COL].astype(str).str.contains(search, na=False)
        ]

    if filtered_latest.empty:
        st.warning("No patients match that search.")
        return

    display_cols = [
        GROUP_COL, time_col, "HR", "MAP", "Temp", "O2Sat", "risk_score", "alert"
    ]

    st.dataframe(
        filtered_latest[display_cols].sort_values("risk_score", ascending=False),
        use_container_width=True
    )



    # The selected patient is shown in more detail below the table.
    selected_patient = st.selectbox(
        "Select patient to view details",
        filtered_latest[GROUP_COL].values
    )

    if selected_patient is not None:

        patient_data = (
            df[df[GROUP_COL] == selected_patient]
            .sort_values(time_col)
            .copy()
        )

        latest_row = patient_data.tail(1)
        latest_x_row = demo_X.loc[latest_row.index, bundle["feature_columns"]]

        risk = float(latest_row["risk_score"].iloc[0])
        latest_state = int(latest_row["alert_state"].iloc[0])

        current_observation_time = latest_row[time_col].iloc[0]
        current_ack = get_acknowledgement_for_alert(
            selected_patient,
            current_observation_time
        )

        col1, col2 = st.columns(2)

        col1.metric("Current risk score", f"{risk:.2f}")
        col2.metric(
            "Alert status",
            "🔴 ALERT" if latest_state == 2 else "🟠 WATCH" if latest_state == 1 else "🟢 OK"
        )

        st.subheader("Latest vitals")

        st.dataframe(
            patient_data[[time_col, "HR", "MAP", "Temp", "O2Sat"]]
            .sort_values(time_col, ascending=False)
            .head(10),
            use_container_width=True
        )

        st.subheader("Risk trend")

        trend = patient_data.sort_values(time_col)
        st.line_chart(trend.set_index(time_col)["risk_score"])

        st.subheader("Why the model flagged this patient")

        st.subheader("Alert acknowledgement")



        # Only show the acknowledgement button when there is an active alert.
        if latest_state == 0:
            st.info("No active alert to acknowledge for the latest observation.")
        else:
            ack_note = st.text_area(
                "Optional acknowledgement note",
                placeholder="e.g. reviewed vitals, contacted senior clinician, repeat observations requested",
                key=f"ack_note_{selected_patient}_{current_observation_time}",
                height=80
            )

            if current_ack is None:
                if st.button("Acknowledge current alert", type="primary"):
                    append_audit_event(
                        action="alert_acknowledged",
                        patient_id=selected_patient,
                        observation_time=current_observation_time,
                        alert_state=latest_state,
                        risk_score=risk,
                        details={
                            "alert_label": alert_state_to_label(latest_state),
                            "time_col": time_col,
                            "note": ack_note.strip(),
                        },
                    )
                    st.success("Alert acknowledged and written to audit log.")
                    st.rerun()
            else:
                ack_details = parse_audit_details(current_ack.get("details", ""))
                ack_time = current_ack.get("timestamp_utc")

                if pd.notna(ack_time):
                    ack_time_text = pd.Timestamp(ack_time).strftime("%Y-%m-%d %H:%M:%S UTC")
                else:
                    ack_time_text = "unknown time"

                st.success(
                    f"Acknowledged by {current_ack.get('user_email', 'unknown user')} at {ack_time_text}"
                )

                if ack_details.get("note"):
                    st.caption(f"Note: {ack_details['note']}")




        # Show recent audit events for this specific patient.
        patient_audit = load_audit_log().copy()
        patient_audit = patient_audit[
            patient_audit["patient_id"].astype(str) == normalise_key(selected_patient)
        ].sort_values("timestamp_utc", ascending=False)

        if not patient_audit.empty:
            patient_audit["timestamp_utc"] = patient_audit["timestamp_utc"].dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            patient_audit["alert_label"] = patient_audit["alert_state"].apply(alert_state_to_label)

            st.subheader("Patient audit trail")

            st.dataframe(
                patient_audit[
                    [
                        "timestamp_utc",
                        "user_email",
                        "role",
                        "action",
                        "observation_time",
                        "alert_label",
                        "risk_score",
                        "details",
                    ]
                ].head(10),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Explanation method: Tree SHAP for Random Forest, "
            "direct feature contribution for Logistic Regression."
        )



        # Calculate the top reasons behind the selected patient's latest prediction.
        explanation_df = compute_local_explanation_df(bundle, latest_x_row)
        top_explanations = explanation_df.head(6).copy()

        st.caption(
            f"Explanation method used: {top_explanations['method'].iloc[0]} | "
            f"Model: {top_explanations['model_type'].iloc[0]}"
        )

        for _, row in top_explanations.iterrows():
            arrow = "↑" if row["contribution"] >= 0 else "↓"
            st.write(f"**{row['feature']}**: {arrow} {row['effect']}")

        st.dataframe(
            top_explanations[["feature", "contribution", "effect"]],
            hide_index=True,
            use_container_width=True
        )



        # Plot the explanation as a simple bar chart.
        plot_df = top_explanations.sort_values("contribution")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(plot_df["feature"], plot_df["contribution"])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Local contribution to prediction")
        ax.set_ylabel("")
        ax.set_title("Top drivers of the current prediction")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)




# ADMIN PAGE 

# This page allows an admin user to adjust the amber and red thresholds.
def admin_page():

    st.title("ICU Sepsis Early Warning System")
    st.header("Admin Threshold Settings")

    bundle = load_bundle()
    df, demo_X = load_demo_assets()

    df = df.copy()
    demo_X = demo_X.copy()

    time_col = get_time_col(df)
    policy = get_alert_policy(bundle)

    default_amber = float(bundle["thresholds"]["amber"])
    default_red = float(bundle["thresholds"]["red"])

    st.write(
        f"Model defaults: amber={default_amber:.4f}, red={default_red:.4f}"
    )



    # Show whether the admin has already saved custom thresholds.
    persisted = load_persisted_thresholds()
    if persisted is None:
        st.info("No saved threshold override found. App starts from model defaults.")
    else:
        st.success(
            f"Saved override on disk: amber={persisted['amber']:.4f}, red={persisted['red']:.4f}"
        )
        if persisted.get("saved_at"):
            st.caption(f"Last saved: {persisted['saved_at']}")

    st.caption(
        "Moving the sliders updates the current session immediately. "
        "Click Save thresholds to keep them after an app restart."
    )

    amber_thr = st.slider(
        "Amber threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.amber_threshold),
        step=0.001
    )

    red_thr = st.slider(
        "Red threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.red_threshold),
        step=0.001
    )




    # Prevent an invalid threshold setup.
    if amber_thr > red_thr:
        st.warning("Amber cannot be above red. Amber has been clipped to red for this session.")
        amber_thr = red_thr

    st.session_state.amber_threshold = float(amber_thr)
    st.session_state.red_threshold = float(red_thr)

    b1, b2, b3 = st.columns([1.3, 1.3, 2])

    if b1.button("Save thresholds"):
        try:
            save_persisted_thresholds(
                st.session_state.amber_threshold,
                st.session_state.red_threshold
            )

            append_audit_event(
                action="threshold_saved",
                details={
                    "amber": float(st.session_state.amber_threshold),
                    "red": float(st.session_state.red_threshold),
                }
            )

            st.success("Thresholds saved. Future app restarts will load these values.")
        except Exception as e:
            st.error(f"Could not save thresholds: {e}")

    if b2.button("Reset to model defaults"):
        clear_persisted_thresholds()
        reset_thresholds_to_defaults(bundle)

        append_audit_event(
            action="threshold_reset_to_default",
            details={
                "amber": float(bundle["thresholds"]["amber"]),
                "red": float(bundle["thresholds"]["red"]),
            }
        )

        st.rerun()

    b3.metric(
        "Active thresholds",
        f"A {st.session_state.amber_threshold:.3f} | R {st.session_state.red_threshold:.3f}"
    )

    st.caption(
        f"Current policy: k_amber={policy['k_amber']}, "
        f"k_red={policy['k_red']}, cooldown={policy['cooldown_hours']}h"
    )




    # Recalculate dashboard impact using the active thresholds.
    df["risk_score"] = bundle["base_model"].predict_proba(
        demo_X[bundle["feature_columns"]]
    )[:, 1]

    alert_df = apply_alert_hygiene(
        df[[GROUP_COL, time_col, "risk_score"]].copy(),
        patient_col=GROUP_COL,
        time_col=time_col,
        score_col="risk_score",
        thr_amber=st.session_state.amber_threshold,
        thr_red=st.session_state.red_threshold,
        k_amber=policy["k_amber"],
        k_red=policy["k_red"],
        cooldown_hours=policy["cooldown_hours"],
        allow_escalation=policy["allow_escalation"],
    )

    df["alert_state"] = alert_df["alert_state"].values

    latest = (
        df.sort_values(time_col)
        .groupby(GROUP_COL)
        .tail(1)
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Patients monitored", len(latest))
    m2.metric("Patients in RED alert", int((latest["alert_state"] == 2).sum()))
    m3.metric("Patients in AMBER alert", int((latest["alert_state"] == 1).sum()))




# Manager page


# This page shows model performance and confusion matrix results.
def manager_page():

    st.title("ICU Sepsis Early Warning System")
    st.header("Manager Performance Dashboard")

    bundle = load_bundle()
    results_df = pd.read_csv(DATA_DIR / "model_results.csv")

    amber_thr, red_thr = get_active_thresholds()
    policy = get_alert_policy(bundle)

    st.subheader("Model evaluation results")
    st.caption(
        "The table below shows the saved export metrics from training time. "
        "The confusion matrix below is recalculated using the active thresholds in this session."
    )
    st.dataframe(results_df, use_container_width=True)

    col1, col2, col3 = st.columns([1, 1, 3])

    if "ROC_AUC" in results_df.columns:
        col1.metric("ROC-AUC", round(results_df["ROC_AUC"].iloc[-1], 3))

    if "PR_AUC" in results_df.columns:
        col2.metric("PR-AUC", round(results_df["PR_AUC"].iloc[-1], 3))

    col3.metric(
        "Active thresholds",
        f"A {amber_thr:.3f} | R {red_thr:.3f}"
    )

    eval_df = load_eval_frame().copy()
    time_col = get_time_col(eval_df)



    # Reapply the alert logic so the manager view reflects the current thresholds.
    eval_alerts = apply_alert_hygiene(
        eval_df[[GROUP_COL, time_col, "risk_score"]].copy(),
        patient_col=GROUP_COL,
        time_col=time_col,
        score_col="risk_score",
        thr_amber=amber_thr,
        thr_red=red_thr,
        k_amber=policy["k_amber"],
        k_red=policy["k_red"],
        cooldown_hours=policy["cooldown_hours"],
        allow_escalation=policy["allow_escalation"],
    )

    eval_df["alert_state"] = eval_alerts["alert_state"].values
    eval_df["alert_fired"] = eval_alerts["alert_fired"].values


    st.subheader("Confusion Matrix")
    st.write(f"Held-out test set ({len(eval_df):,} hourly records)")



    # Allow the manager to compare raw model output against the alert-hygiene version.
    view = st.radio(
        "Evaluation view",
        ["Raw RED threshold", "RED alert state (with hygiene)", "ANY alert state (with hygiene)"],
        horizontal=True
    )

    y_true = eval_df["y_true"].astype(int)

    if view == "Raw RED threshold":
        y_pred = (eval_df["risk_score"] >= red_thr).astype(int)
        st.caption(f"Raw hourly predictions at RED threshold = {red_thr:.4f}")
    elif view == "RED alert state (with hygiene)":
        y_pred = (eval_df["alert_state"] == 2).astype(int)
        st.caption("RED alert state after cooldown / alert-hygiene logic")
    else:
        y_pred = (eval_df["alert_state"] >= 1).astype(int)
        st.caption("ANY alert state (amber or red) after cooldown / alert-hygiene logic")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(2, 2), dpi=120)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No EW", "EW"]
    )

    disp.plot(
        ax=ax,
        cmap="RdPu",
        colorbar=False,
        values_format="d"
    )

    ax.set_xlabel("Predicted", fontsize=5, labelpad=1)
    ax.set_ylabel("Actual", fontsize=5, labelpad=1)
    ax.tick_params(axis="both", labelsize=4)

    for text in ax.texts:
        text.set_fontsize(5)

    fig.tight_layout(pad=0.1)

    left, mid, right = st.columns([2, 1.2, 2])
    with mid:
        st.pyplot(fig, use_container_width=False, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TN", f"{tn:,}")
    c2.metric("FP", f"{fp:,}")
    c3.metric("FN", f"{fn:,}")
    c4.metric("TP", f"{tp:,}")




# Audit log page

# This page lets admins and managers view and download recorded actions.
def audit_log_page():

    st.title("ICU Sepsis Early Warning System")
    st.header("Audit Log")

    st.caption(
        "Prototype note: this is a CSV-based audit trail for demonstration, testing, "
        "and accountability. It is not a tamper-proof production audit system."
    )

    audit_df = load_audit_log().copy()

    if audit_df.empty:
        st.info("No audit events recorded yet.")
        return

    audit_df = audit_df.sort_values("timestamp_utc", ascending=False)
    audit_df["timestamp_display"] = audit_df["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    audit_df["alert_label"] = audit_df["alert_state"].apply(alert_state_to_label)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total events", len(audit_df))
    c2.metric(
        "Alert acknowledgements",
        int((audit_df["action"].astype(str) == "alert_acknowledged").sum())
    )
    c3.metric(
        "Threshold changes",
        int(
            audit_df["action"].astype(str).isin(
                ["threshold_saved", "threshold_reset_to_default"]
            ).sum()
        )
    )


    # Filters make the audit log easier to inspect.
    action_options = ["All"] + sorted(audit_df["action"].dropna().astype(str).unique().tolist())
    action_filter = st.selectbox("Filter by action", action_options)

    patient_filter = st.text_input("Filter by patient ID")

    filtered = audit_df.copy()

    if action_filter != "All":
        filtered = filtered[filtered["action"].astype(str) == action_filter]

    if patient_filter:
        filtered = filtered[
            filtered["patient_id"].astype(str).str.contains(patient_filter, na=False)
        ]

    st.dataframe(
        filtered[
            [
                "timestamp_display",
                "user_email",
                "role",
                "action",
                "patient_id",
                "observation_time",
                "alert_label",
                "risk_score",
                "details",
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download audit log CSV",
        data=filtered.to_csv(index=False),
        file_name="audit_log.csv",
        mime="text/csv"
    )




# Page routing

# The role controls which pages the user can access.
if not st.session_state.logged_in:

    login_page()

else:

    role = st.session_state.role

    st.sidebar.title("Navigation")
    st.sidebar.write(f"Logged in as: {role}")

    if role == "Clinician":
        page = st.sidebar.radio("Go to", ["Dashboard"])

    elif role == "Admin":
        page = st.sidebar.radio("Go to", ["Admin Settings", "Audit Log"])

    elif role == "Manager":
        page = st.sidebar.radio("Go to", ["Performance", "Audit Log"])

    if page == "Dashboard":
        clinician_dashboard()

    elif page == "Admin Settings":
        admin_page()

    elif page == "Performance":
        manager_page()

    elif page == "Audit Log":
        audit_log_page()



    # References
    # These links record the main documentation pages and sources used while building the Streamlit prototype.
    # Project-specific logic, including the role structure, dashboard workflow, alert hygiene rules,
    # threshold handling, audit actions and page routing, was designed, implemented and tested within this project.

    # Project-specific implementation:
    # P0 = Project-specific dashboard design and implementation decisions, including role-based users,
    # clinician/admin/manager workflows, alert-state logic, threshold persistence, acknowledgement logging,
    # audit-log structure and dashboard layout. These were designed and validated within this project.

    # Streamlit documentation:
    # Streamlit developers (n.d.) Streamlit API reference. Available at: https://docs.streamlit.io/develop/api-reference
    # Streamlit developers (n.d.) st.set_page_config. Available at: https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
    # Streamlit developers (n.d.) Session State. Available at: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    # Streamlit developers (n.d.) st.rerun. Available at: https://docs.streamlit.io/develop/api-reference/execution-flow/st.rerun
    # Streamlit developers (n.d.) st.sidebar. Available at: https://docs.streamlit.io/develop/api-reference/layout/st.sidebar
    # Streamlit developers (n.d.) st.columns. Available at: https://docs.streamlit.io/develop/api-reference/layout/st.columns
    # Streamlit developers (n.d.) st.title. Available at: https://docs.streamlit.io/develop/api-reference/text/st.title
    # Streamlit developers (n.d.) st.header. Available at: https://docs.streamlit.io/develop/api-reference/text/st.header
    # Streamlit developers (n.d.) st.subheader. Available at: https://docs.streamlit.io/develop/api-reference/text/st.subheader
    # Streamlit developers (n.d.) st.caption. Available at: https://docs.streamlit.io/develop/api-reference/text/st.caption
    # Streamlit developers (n.d.) st.write. Available at: https://docs.streamlit.io/develop/api-reference/write-magic/st.write
    # Streamlit developers (n.d.) st.info. Available at: https://docs.streamlit.io/develop/api-reference/status/st.info
    # Streamlit developers (n.d.) st.warning. Available at: https://docs.streamlit.io/develop/api-reference/status/st.warning
    # Streamlit developers (n.d.) st.error. Available at: https://docs.streamlit.io/develop/api-reference/status/st.error
    # Streamlit developers (n.d.) st.success. Available at: https://docs.streamlit.io/develop/api-reference/status/st.success
    # Streamlit developers (n.d.) st.button. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.button
    # Streamlit developers (n.d.) st.text_input. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.text_input
    # Streamlit developers (n.d.) st.text_area. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.text_area
    # Streamlit developers (n.d.) st.selectbox. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox
    # Streamlit developers (n.d.) st.slider. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.slider
    # Streamlit developers (n.d.) st.radio. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.radio
    # Streamlit developers (n.d.) st.metric. Available at: https://docs.streamlit.io/develop/api-reference/data/st.metric
    # Streamlit developers (n.d.) st.dataframe. Available at: https://docs.streamlit.io/develop/api-reference/data/st.dataframe
    # Streamlit developers (n.d.) st.line_chart. Available at: https://docs.streamlit.io/develop/api-reference/charts/st.line_chart
    # Streamlit developers (n.d.) st.pyplot. Available at: https://docs.streamlit.io/develop/api-reference/charts/st.pyplot
    # Streamlit developers (n.d.) st.download_button. Available at: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button

    # Python standard library:
    # Python Software Foundation (n.d.) The Python Language Reference. Available at: https://docs.python.org/3/reference/index.html
    # Python Software Foundation (n.d.) Built-in functions. Available at: https://docs.python.org/3/library/functions.html
    # Python Software Foundation (n.d.) pathlib — Object-oriented filesystem paths. Available at: https://docs.python.org/3/library/pathlib.html
    # Python Software Foundation (n.d.) json — JSON encoder and decoder. Available at: https://docs.python.org/3/library/json.html

    # pandas documentation:
    # The pandas development team (n.d.) pandas.DataFrame. Available at: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    # The pandas development team (n.d.) pandas.Series. Available at: https://pandas.pydata.org/docs/reference/api/pandas.Series.html
    # The pandas development team (n.d.) Input/output. Available at: https://pandas.pydata.org/docs/user_guide/io.html
    # The pandas development team (n.d.) pandas.read_csv. Available at: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    # The pandas development team (n.d.) pandas.read_parquet. Available at: https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
    # The pandas development team (n.d.) pandas.DataFrame.to_csv. Available at: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    # The pandas development team (n.d.) pandas.Timestamp.utcnow. Available at: https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.utcnow.html
    # The pandas development team (n.d.) pandas.to_datetime. Available at: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
    # The pandas development team (n.d.) pandas.isna. Available at: https://pandas.pydata.org/docs/reference/api/pandas.isna.html
    # The pandas development team (n.d.) pandas.notna. Available at: https://pandas.pydata.org/docs/reference/api/pandas.notna.html
    # The pandas development team (n.d.) GroupBy. Available at: https://pandas.pydata.org/docs/reference/groupby.html
    # The pandas development team (n.d.) pandas.Series.str.contains. Available at: https://pandas.pydata.org/docs/reference/api/pandas.Series.str.contains.html
    # The pandas development team (n.d.) pandas.Series.dt.strftime. Available at: https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.strftime.html

    # NumPy documentation:
    # NumPy Developers (n.d.) NumPy documentation. Available at: https://numpy.org/doc/stable/
    # NumPy Developers (n.d.) numpy.asarray. Available at: https://numpy.org/doc/stable/reference/generated/numpy.asarray.html
    # NumPy Developers (n.d.) numpy.zeros. Available at: https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    # NumPy Developers (n.d.) numpy.where. Available at: https://numpy.org/doc/stable/reference/generated/numpy.where.html
    # NumPy Developers (n.d.) numpy.select. Available at: https://numpy.org/doc/stable/reference/generated/numpy.select.html
    # NumPy Developers (n.d.) NumPy scalar types. Available at: https://numpy.org/doc/stable/reference/arrays.scalars.html

    # joblib documentation:
    # Joblib developers (n.d.) joblib.load. Available at: https://joblib.readthedocs.io/en/latest/generated/joblib.load.html

    # SHAP documentation:
    # SHAP contributors (n.d.) SHAP documentation. Available at: https://shap.readthedocs.io/en/latest/
    # SHAP contributors (n.d.) shap.TreeExplainer. Available at: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
    # SHAP contributors (n.d.) shap.Explainer. Available at: https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html

    # scikit-learn documentation:
    # scikit-learn developers (n.d.) Pipeline. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    # scikit-learn developers (n.d.) LogisticRegression. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # scikit-learn developers (n.d.) RandomForestClassifier. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # scikit-learn developers (n.d.) confusion_matrix. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # scikit-learn developers (n.d.) ConfusionMatrixDisplay. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

    # Matplotlib documentation:
    # The Matplotlib Development Team (n.d.) matplotlib.pyplot. Available at: https://matplotlib.org/stable/api/pyplot_summary.html
    # The Matplotlib Development Team (n.d.) matplotlib.pyplot.subplots. Available at: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    # The Matplotlib Development Team (n.d.) matplotlib.pyplot.close. Available at: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html
    # The Matplotlib Development Team (n.d.) Axes API. Available at: https://matplotlib.org/stable/api/axes_api.html

    # SciPy documentation:
    # SciPy developers (n.d.) scipy.sparse.csr_matrix.toarray. Available at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.toarray.html