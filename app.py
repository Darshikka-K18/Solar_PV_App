"""
Solar PV Diagnostic App
------------------------
One upload button -> auto-routes to CNN / Random Forest / LSTM ->
CrewAI agent turns the raw prediction into a plain-English maintenance ticket.

Run with:  streamlit run app.py
"""

import os
import json
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from crewai import Agent, Task, Crew, LLM

# On Streamlit Community Cloud, secrets are set in the app's Settings > Secrets
# (TOML format) rather than a local .env file. This makes these keys work in
# both environments without changing any other code.
for _key in ("OPENAI_API_KEY", "GROQ_API_KEY"):
    try:
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
    except Exception:
        pass  # no secrets.toml present (e.g. running purely locally) -- fine


def get_agent_llm():
    """Prefer Groq (free tier) if a key is set; otherwise fall back to OpenAI."""
    if os.environ.get("GROQ_API_KEY"):
        return LLM(model="groq/llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])
    if os.environ.get("OPENAI_API_KEY"):
        return LLM(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    return None

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

MODELS_DIR = "models"

CNN_MODEL_PATH = os.path.join(MODELS_DIR, "CNN_PV_Fault_Model.keras")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")

RF_MODEL_1_PATH = os.path.join(MODELS_DIR, "rf_model_1_string.pkl")
RF_SCALER_1_PATH = os.path.join(MODELS_DIR, "rf_scaler_1_string.pkl")
RF_MODEL_3_PATH = os.path.join(MODELS_DIR, "rf_model_3_string.pkl")
RF_SCALER_3_PATH = os.path.join(MODELS_DIR, "rf_scaler_3_string.pkl")

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "LSTM_RUL_Model.keras")
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
LSTM_WINDOW = 12  # must match training script

# RF models predict an integer-coded Label (0-4). This order matches the
# class_names list used when the confusion matrix was generated in training.
RF_CLASS_NAMES = ['Normal', 'Shading', 'Short', 'Connector', 'OC']

# --------------------------------------------------------------------------
# CACHED MODEL LOADERS (loaded once, reused across requests)
# --------------------------------------------------------------------------

@st.cache_resource
def load_cnn():
    model = load_model(CNN_MODEL_PATH)
    with open(CLASS_MAPPING_PATH) as f:
        class_mapping = json.load(f)
    return model, class_mapping


@st.cache_resource
def load_rf():
    model_1 = joblib.load(RF_MODEL_1_PATH)
    scaler_1 = joblib.load(RF_SCALER_1_PATH)
    model_3 = joblib.load(RF_MODEL_3_PATH)
    scaler_3 = joblib.load(RF_SCALER_3_PATH)
    return (model_1, scaler_1), (model_3, scaler_3)


@st.cache_resource
def load_lstm():
    model = load_model(LSTM_MODEL_PATH)
    scaler = joblib.load(LSTM_SCALER_PATH)
    return model, scaler


# --------------------------------------------------------------------------
# STEP 2: ROUTER
# --------------------------------------------------------------------------

def route_file(uploaded_file):
    """Return one of: 'image', 'rf', 'lstm', or raise ValueError."""
    name = uploaded_file.name.lower()

    if name.endswith((".jpg", ".jpeg", ".png")):
        return "image"

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # reset pointer so it can be read again downstream
        if len(df) == 1:
            return "rf"
        elif len(df) > 1:
            return "lstm"
        else:
            raise ValueError("CSV file appears to be empty.")

    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


# --------------------------------------------------------------------------
# INFERENCE: CNN (image fault detection)
# --------------------------------------------------------------------------

def run_cnn_inference(uploaded_file):
    model, class_mapping = load_cnn()

    # Derive expected input size directly from the model, so we don't hardcode it.
    _, target_h, target_w, channels = model.input_shape

    img = Image.open(uploaded_file).convert("RGB" if channels == 3 else "L")
    img = img.resize((target_w, target_h))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx]) * 100

    # class_mapping.json was saved from Keras's `train_data.class_indices`,
    # which is {label_name: index} -- so we invert it to look up by index.
    index_to_label = {v: k for k, v in class_mapping.items()}
    label = index_to_label.get(class_idx, f"Class_{class_idx}")

    return {
        "data_type": "RGB Image",
        "detection": label,
        "confidence": round(confidence, 2),
    }


# --------------------------------------------------------------------------
# INFERENCE: Random Forest (single-row sensor snapshot)
# --------------------------------------------------------------------------

def run_rf_inference_from_df(df: pd.DataFrame, string_config: str):
    """string_config must be '1-string' or '3-string' (chosen by the user)."""
    (model_1, scaler_1), (model_3, scaler_3) = load_rf()

    if string_config == "1-string":
        model, scaler = model_1, scaler_1
    elif string_config == "3-string":
        model, scaler = model_3, scaler_3
    else:
        raise ValueError(f"Unknown string_config: {string_config}")

    # Each scaler remembers the exact feature names/order it was fit on
    # (sklearn stores this as feature_names_in_ when fit on a DataFrame).
    missing = set(scaler.feature_names_in_) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data is missing columns required by the {string_config} model: {sorted(missing)}\n"
            f"Expected columns: {list(scaler.feature_names_in_)}"
        )

    # Reorder/select columns to exactly match what the scaler was fit on.
    X = df[scaler.feature_names_in_].values
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba_vec = None
    if hasattr(model, "predict_proba"):
        proba_vec = model.predict_proba(X_scaled)[0]

    # Label was integer-coded (0-4) during training; map back to a fault name.
    try:
        label = RF_CLASS_NAMES[int(pred)]
    except (ValueError, IndexError, TypeError):
        label = str(pred)  # fall back to raw value if it wasn't an int 0-4

    confidence = round(float(np.max(proba_vec)) * 100, 2) if proba_vec is not None else "N/A"

    result = {
        "data_type": f"Sensor Snapshot ({string_config} config)",
        "detection": label,
        "confidence": confidence,
    }

    # Extra debug info -- shown in an expander in the UI, useful for
    # diagnosing "this model gives weird results" type issues.
    if proba_vec is not None:
        result["_debug_class_probabilities"] = {
            RF_CLASS_NAMES[i] if i < len(RF_CLASS_NAMES) else str(i): round(float(p), 4)
            for i, p in enumerate(proba_vec)
        }
    result["_debug_features_used"] = {
        col: float(val) for col, val in zip(scaler.feature_names_in_, X[0])
    }

    return result


def run_rf_inference(uploaded_file, string_config: str):
    df = pd.read_csv(uploaded_file)
    return run_rf_inference_from_df(df, string_config)


# --------------------------------------------------------------------------
# INFERENCE: LSTM (time-series RUL projection)
# --------------------------------------------------------------------------

def run_lstm_inference(uploaded_file):
    model, scaler = load_lstm()
    df = pd.read_csv(uploaded_file)

    # Assumes a 'Health_Indicator' column, same as training. Adjust if your
    # production CSVs use a different column name.
    if "Health_Indicator" not in df.columns:
        raise ValueError("Expected a 'Health_Indicator' column in the time-series CSV.")

    values = df["Health_Indicator"].values.reshape(-1, 1)
    if len(values) < LSTM_WINDOW:
        raise ValueError(f"Need at least {LSTM_WINDOW} rows of history for a projection.")

    scaled = scaler.transform(values)
    threshold = df["Health_Indicator"].iloc[0] * 0.8

    current_batch = scaled[-LSTM_WINDOW:].reshape(1, LSTM_WINDOW, 1)
    future_projections = []
    for _ in range(24):
        current_pred = model.predict(current_batch, verbose=0)
        future_projections.append(current_pred[0, 0])
        current_batch = np.append(current_batch[:, 1:, :], [current_pred], axis=1)

    projected_vals = scaler.inverse_transform(np.array(future_projections).reshape(-1, 1))
    failure_indices = np.where(projected_vals < threshold)[0]

    # Try to build real calendar dates if a timestamp column exists; otherwise use month offsets.
    if "timestamp" in df.columns:
        last_date = pd.to_datetime(df["timestamp"]).iloc[-1]
    else:
        last_date = pd.Timestamp.today()

    projection_dates = pd.date_range(start=last_date + timedelta(days=30), periods=24, freq="ME")

    if len(failure_indices) > 0:
        failure_date = projection_dates[failure_indices[0]].strftime("%Y-%m-%d")
        detection = f"Projected failure around {failure_date}"
    else:
        failure_date = None
        detection = "No failure projected within the next 2 years"

    return {
        "data_type": "Time-Series Sensor Log",
        "detection": detection,
        "confidence": "N/A",
        "current_health": round(float(df["Health_Indicator"].iloc[-1]), 2),
        "threshold": round(float(threshold), 2),
        "failure_date": failure_date,
    }


# --------------------------------------------------------------------------
# STEP 3: AGENTIC AI LAYER (CrewAI)
# --------------------------------------------------------------------------

def generate_maintenance_report(prediction: dict) -> str:
    """Pass the raw model output to a single CrewAI agent and get back a
    plain-English maintenance ticket."""

    agent_llm = get_agent_llm()
    if agent_llm is None:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (free) or OPENAI_API_KEY "
            "as an environment variable or Streamlit secret."
        )

    technician = Agent(
        role="Senior Solar Maintenance Technician",
        goal=(
            "Translate raw solar PV diagnostic model output into a clear, "
            "actionable maintenance ticket a non-technical client can understand."
        ),
        backstory=(
            "You have 15 years of field experience maintaining utility-scale and "
            "residential solar arrays. You write concise, professional maintenance "
            "notes that always include a concrete recommended action and timeframe."
        ),
        llm=agent_llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "Here is the raw diagnostic output from our fault-detection pipeline:\n\n"
            f"{json.dumps(prediction, indent=2)}\n\n"
            "Write a short 'Technician Notes' section (2-4 sentences) explaining, in "
            "plain English, what was detected, why it matters, and a concrete "
            "recommended action with a timeframe. Do not repeat the raw JSON."
        ),
        expected_output="A 2-4 sentence maintenance note in plain English.",
        agent=technician,
    )

    crew = Crew(agents=[technician], tasks=[task], verbose=False)
    result = crew.kickoff()
    return str(result)


# --------------------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------------------

def show_report(prediction: dict):
    st.subheader("Diagnostic Report")
    st.write(f"**Data Processed:** {prediction['data_type']}")
    st.write(f"**Detection:** {prediction['detection']}"
             + (f" ({prediction['confidence']}% Confidence)"
                if prediction.get("confidence") not in (None, "N/A") else ""))

    # Debug info, if present (RF predictions include this) -- helps diagnose
    # cases where a model seems to give wrong/unexpected results.
    if "_debug_class_probabilities" in prediction or "_debug_features_used" in prediction:
        with st.expander("Debug details"):
            if "_debug_features_used" in prediction:
                st.write("**Feature values sent to the model (in order):**")
                st.json(prediction["_debug_features_used"])
            if "_debug_class_probabilities" in prediction:
                st.write("**Class probabilities:**")
                st.json(prediction["_debug_class_probabilities"])

    with st.spinner("Agent is drafting the technician note..."):
        try:
            clean_prediction = {k: v for k, v in prediction.items() if not k.startswith("_debug")}
            note = generate_maintenance_report(clean_prediction)
            st.write("**Technician Notes (Agent AI):**")
            st.info(note)
        except Exception as e:
            st.warning(
                f"Agent report generation failed ({e}). Showing raw prediction only."
            )


def manual_entry_tab():
    st.write("Manually key in a single sensor reading (no CSV needed).")

    string_config = st.radio(
        "How many strings does this inverter/array have?",
        ["1-string", "3-string"],
        horizontal=True,
        key="manual_string_config",
    )

    (model_1, scaler_1), (model_3, scaler_3) = load_rf()
    scaler = scaler_1 if string_config == "1-string" else scaler_3
    feature_names = list(scaler.feature_names_in_)

    st.caption(f"This model expects {len(feature_names)} value(s): {', '.join(feature_names)}")

    values = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            values[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"manual_{feat}")

    if st.button("Run Diagnostic", type="primary"):
        df = pd.DataFrame([values])
        with st.spinner("Running diagnostic model..."):
            try:
                prediction = run_rf_inference_from_df(df, string_config)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return
        show_report(prediction)


def upload_tab():
    uploaded_file = st.file_uploader(
        "Upload file", type=["png", "jpg", "jpeg", "csv"]
    )

    if uploaded_file is None:
        st.info("Waiting for a file...")
        return

    try:
        route = route_file(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return

    string_config = None
    if route == "rf":
        string_config = st.radio(
            "This looks like a single sensor snapshot. How many strings does "
            "this inverter/array have?",
            ["1-string", "3-string"],
            horizontal=True,
        )

    with st.spinner("Running diagnostic model..."):
        try:
            if route == "image":
                prediction = run_cnn_inference(uploaded_file)
            elif route == "rf":
                prediction = run_rf_inference(uploaded_file, string_config)
            elif route == "lstm":
                prediction = run_lstm_inference(uploaded_file)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    show_report(prediction)


def main():
    st.set_page_config(page_title="Solar PV Diagnostic Assistant", page_icon="☀️")
    st.title("☀️ Solar PV Diagnostic Assistant")
    st.write(
        "Upload a panel photo / CSV, or manually type in a single sensor "
        "reading. The app figures out which model to run."
    )

    tab1, tab2 = st.tabs(["📁 Upload File", "⌨️ Manual Entry (RF)"])
    with tab1:
        upload_tab()
    with tab2:
        manual_entry_tab()


if __name__ == "__main__":
    main()
