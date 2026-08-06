import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from crewai import Agent, Task, Crew, LLM

from design import inject_theme, render_hero, show_report as render_ticket, render_technician_notes

# --------------------------------------------------------------------------
# SECRETS (Streamlit Community Cloud sets these under Settings > Secrets)
# --------------------------------------------------------------------------
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


def get_agent_llm():
    # Workaround for a CrewAI bug: it injects a `cache_breakpoint` marker meant
    # for Anthropic-style prompt caching but doesn't strip it for other
    # providers, so Groq rejects the request. See crewAI issue #5886.
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg

    return LLM(model="groq/llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])


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

# RF models predict an integer-coded label (0-4); order matches training.
RF_CLASS_NAMES = ['Normal', 'Shading', 'Short', 'Connector', 'OC']

# --------------------------------------------------------------------------
# ARRAY-CONFIG + PANEL-TYPE SCALING
# --------------------------------------------------------------------------
# The RF models were trained on one specific simulated panel (66 cells,
# Voc 47.42V, Isc 15A, Vmp 39.51V, Imp 14.17A), wired 7 in series and either
# 1 or 3 strings in parallel. Two independent things can mismatch a real
# reading, and both get corrected before scoring:
#   1. ARRAY WIRING -- voltage scales with series count, current scales with
#      parallel count, power scales with both.
#   2. PANEL TYPE -- a different physical panel. We normalize the reading to
#      a fraction of the user's own panel ratings, then re-scale that
#      fraction onto the reference panel's ratings (assumes similar IV-curve
#      shape; doesn't correct for differing temperature coefficients).
# Temperature/irradiance readings are left unchanged by either correction.

REFERENCE_PANELS_IN_SERIES = 7  # panels per string the models were trained on

REFERENCE_PARALLEL_STRINGS = {
    "1-string": 1,
    "3-string": 3,
}

REFERENCE_PANEL_SPECS = {
    "Voc": 47.42,   # open-circuit voltage (V)
    "Isc": 15.0,    # short-circuit current (A)
    "Vmp": 39.51,   # voltage at max power point (V)
    "Imp": 14.17,   # current at max power point (A)
}
REFERENCE_PANEL_SPECS["Pmax"] = REFERENCE_PANEL_SPECS["Vmp"] * REFERENCE_PANEL_SPECS["Imp"]

# Maps each data column to which array dimension it scales with, and which
# nameplate spec it should be normalized against. Temp_C / Irr_Wm2 are
# intentionally absent -- they don't scale with wiring or panel type.
COLUMN_SCALE_CONFIG = {
    "Voc_V":  {"array_dim": "series",   "spec_key": "Voc"},
    "Vmp_V":  {"array_dim": "series",   "spec_key": "Vmp"},
    "Isc_A":  {"array_dim": "parallel", "spec_key": "Isc"},
    "Imp_A":  {"array_dim": "parallel", "spec_key": "Imp"},
    "Pmax_W": {"array_dim": "both",     "spec_key": "Pmax"},
}


def scale_reading(df: pd.DataFrame, panels_in_series: int, parallel_strings: int,
                   reference_parallel_strings: int, user_panel_specs: dict) -> pd.DataFrame:
    """Project a reading from a real array (possibly a different panel model,
    wired differently) onto the reference array/panel the model was trained on."""
    if panels_in_series is None or panels_in_series <= 0:
        raise ValueError("Panels in series must be a positive number.")
    if parallel_strings is None or parallel_strings <= 0:
        raise ValueError("Parallel strings must be a positive number.")

    series_ratio = REFERENCE_PANELS_IN_SERIES / panels_in_series
    parallel_ratio = reference_parallel_strings / parallel_strings
    dim_ratio = {"series": series_ratio, "parallel": parallel_ratio,
                 "both": series_ratio * parallel_ratio}

    user_panel_specs = dict(user_panel_specs or {})
    if "Pmax" not in user_panel_specs and "Vmp" in user_panel_specs and "Imp" in user_panel_specs:
        user_panel_specs["Pmax"] = user_panel_specs["Vmp"] * user_panel_specs["Imp"]

    df = df.copy()
    for col, cfg in COLUMN_SCALE_CONFIG.items():
        if col not in df.columns:
            continue
        array_ratio = dim_ratio[cfg["array_dim"]]

        spec_key = cfg["spec_key"]
        actual_val = user_panel_specs.get(spec_key)
        ref_val = REFERENCE_PANEL_SPECS[spec_key]
        panel_ratio = (ref_val / actual_val) if actual_val else 1.0  # no user spec -> same panel

        df[col] = df[col] * array_ratio * panel_ratio
    return df


# --------------------------------------------------------------------------
# CACHED MODEL LOADERS
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
# ROUTER
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

    _, target_h, target_w, channels = model.input_shape
    img = Image.open(uploaded_file).convert("RGB" if channels == 3 else "L")
    img = img.resize((target_w, target_h))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx]) * 100

    # class_mapping.json was saved from Keras's class_indices ({label: index}).
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

def run_rf_inference_from_df(df: pd.DataFrame, string_config: str,
                              panels_in_series: int = REFERENCE_PANELS_IN_SERIES,
                              parallel_strings: int = None,
                              user_panel_specs: dict = None):
    """string_config: '1-string' or '3-string', picks which trained model to
    use. panels_in_series / parallel_strings describe the actual array wiring
    the reading came from. user_panel_specs (Voc/Isc/Vmp/Imp) describes the
    actual panel model in use; leave None to assume the reference panel."""
    (model_1, scaler_1), (model_3, scaler_3) = load_rf()

    if string_config == "1-string":
        model, scaler = model_1, scaler_1
    elif string_config == "3-string":
        model, scaler = model_3, scaler_3
    else:
        raise ValueError(f"Unknown string_config: {string_config}")

    reference_parallel = REFERENCE_PARALLEL_STRINGS[string_config]
    if parallel_strings is None:
        parallel_strings = reference_parallel

    missing = set(scaler.feature_names_in_) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data is missing columns required by the {string_config} model: {sorted(missing)}\n"
            f"Expected columns: {list(scaler.feature_names_in_)}"
        )

    df = scale_reading(df, panels_in_series, parallel_strings, reference_parallel, user_panel_specs)

    X = df[scaler.feature_names_in_].values
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba_vec = None
    if hasattr(model, "predict_proba"):
        proba_vec = model.predict_proba(X_scaled)[0]

    try:
        label = RF_CLASS_NAMES[int(pred)]
    except (ValueError, IndexError, TypeError):
        label = str(pred)

    confidence = round(float(np.max(proba_vec)) * 100, 2) if proba_vec is not None else "N/A"

    panel_note = "reference panel" if not user_panel_specs else "custom panel, normalized"
    result = {
        "data_type": (
            f"Sensor Snapshot ({string_config} model, "
            f"{panels_in_series} panels/string, {parallel_strings} parallel, {panel_note})"
        ),
        "detection": label,
        "confidence": confidence,
    }

    if proba_vec is not None:
        result["_debug_class_probabilities"] = {
            RF_CLASS_NAMES[i] if i < len(RF_CLASS_NAMES) else str(i): round(float(p), 4)
            for i, p in enumerate(proba_vec)
        }
    result["_debug_features_used"] = {
        col: float(val) for col, val in zip(scaler.feature_names_in_, X[0])
    }

    return result


def run_rf_inference(uploaded_file, string_config: str,
                      panels_in_series: int = REFERENCE_PANELS_IN_SERIES,
                      parallel_strings: int = None,
                      user_panel_specs: dict = None):
    df = pd.read_csv(uploaded_file)
    return run_rf_inference_from_df(df, string_config, panels_in_series, parallel_strings, user_panel_specs)


# --------------------------------------------------------------------------
# INFERENCE: LSTM (time-series RUL projection)
# --------------------------------------------------------------------------

def run_lstm_inference(uploaded_file):
    model, scaler = load_lstm()
    df = pd.read_csv(uploaded_file)

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
# AGENTIC AI LAYER (CrewAI)
# --------------------------------------------------------------------------

def generate_maintenance_report(prediction: dict) -> str:
    """Passes the raw model output to a single CrewAI agent and returns a
    plain-English maintenance ticket."""
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
        llm=get_agent_llm(),
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


def show_report(prediction: dict):
    """Renders the diagnostic ticket, then asks the agent for a technician
    note and renders that underneath."""
    render_ticket(prediction)

    with st.spinner("Agent is drafting the technician note..."):
        try:
            clean_prediction = {k: v for k, v in prediction.items() if not k.startswith("_debug")}
            note = generate_maintenance_report(clean_prediction)
            render_technician_notes(note)
        except Exception as e:
            st.warning(f"Agent report generation failed ({e}). Showing raw prediction only.")


# --------------------------------------------------------------------------
# PANEL SPEC / ARRAY CONFIG UI HELPERS
# --------------------------------------------------------------------------

def panel_spec_inputs(key_prefix: str):
    """Expander asking for the user's actual panel datasheet specs. Returns
    a dict (Voc/Isc/Vmp/Imp) or None if left at defaults (same panel the
    model was trained on)."""
    with st.expander("⚙️ My panel is a different model (optional)"):
        st.caption(
            f"Reference panel this model was trained on: Voc {REFERENCE_PANEL_SPECS['Voc']}V, "
            f"Isc {REFERENCE_PANEL_SPECS['Isc']}A, Vmp {REFERENCE_PANEL_SPECS['Vmp']}V, "
            f"Imp {REFERENCE_PANEL_SPECS['Imp']}A. Enter your panel's own datasheet values "
            "below and readings will be normalized automatically. Leave as-is if you're "
            "using the same panel model."
        )
        c1, c2 = st.columns(2)
        with c1:
            voc = st.number_input("Voc (V)", min_value=0.0, value=REFERENCE_PANEL_SPECS["Voc"],
                                   format="%.2f", key=f"{key_prefix}_voc")
            vmp = st.number_input("Vmp (V)", min_value=0.0, value=REFERENCE_PANEL_SPECS["Vmp"],
                                   format="%.2f", key=f"{key_prefix}_vmp")
        with c2:
            isc = st.number_input("Isc (A)", min_value=0.0, value=REFERENCE_PANEL_SPECS["Isc"],
                                   format="%.2f", key=f"{key_prefix}_isc")
            imp = st.number_input("Imp (A)", min_value=0.0, value=REFERENCE_PANEL_SPECS["Imp"],
                                   format="%.2f", key=f"{key_prefix}_imp")

    specs = {"Voc": voc, "Isc": isc, "Vmp": vmp, "Imp": imp}
    unchanged = all(specs[k] == REFERENCE_PANEL_SPECS[k] for k in specs)
    return None if unchanged else specs


def resolve_string_config(parallel_strings: int) -> str:
    """1 parallel string -> the 1-string model. More than 1 -> the
    multi-string model (its scaling normalizes any actual parallel count)."""
    return "1-string" if parallel_strings == 1 else "3-string"


# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------

def manual_entry_tab():
    st.markdown(
        '<p style="color: var(--text-muted); font-size: 0.92rem;">'
        'Key in a single sensor reading directly — no CSV needed.</p>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        panels_in_series = st.number_input(
            "Panels wired in series per string",
            min_value=1, value=REFERENCE_PANELS_IN_SERIES, step=1,
            key="manual_panels_series",
            help=f"Model reference: {REFERENCE_PANELS_IN_SERIES} panels/string.",
        )
    with col_b:
        parallel_strings = st.number_input(
            "Strings wired in parallel",
            min_value=1, value=1, step=1,
            key="manual_parallel_strings",
            help="1 = single-string site. More than 1 automatically uses the "
                 "multi-string model, normalized to your actual count.",
        )
    string_config = resolve_string_config(parallel_strings)
    st.caption(
        f"Using the **{string_config}** model "
        f"(trained on a {REFERENCE_PARALLEL_STRINGS[string_config]}-parallel-string reference)."
    )

    (model_1, scaler_1), (model_3, scaler_3) = load_rf()
    scaler = scaler_1 if string_config == "1-string" else scaler_3
    feature_names = list(scaler.feature_names_in_)

    st.caption(f"This model expects {len(feature_names)} value(s): {', '.join(feature_names)}")

    user_panel_specs = panel_spec_inputs("manual")

    values = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            values[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"manual_{feat}")

    if st.button("Run Diagnostic", type="primary"):
        df = pd.DataFrame([values])
        with st.spinner("Running diagnostic model..."):
            try:
                prediction = run_rf_inference_from_df(
                    df, string_config, panels_in_series, parallel_strings, user_panel_specs
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return
        show_report(prediction)


def upload_tab():
    uploaded_file = st.file_uploader(
        "Upload file", type=["png", "jpg", "jpeg", "csv"]
    )

    if uploaded_file is None:
        st.info("No file loaded yet — drop an image or CSV above to begin.")
        return

    try:
        route = route_file(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return

    string_config = None
    panels_in_series = REFERENCE_PANELS_IN_SERIES
    parallel_strings = None
    user_panel_specs = None

    if route == "rf":
        st.write("This looks like a single sensor snapshot.")
        col_a, col_b = st.columns(2)
        with col_a:
            panels_in_series = st.number_input(
                "Panels wired in series per string",
                min_value=1, value=REFERENCE_PANELS_IN_SERIES, step=1,
                help=f"Model reference: {REFERENCE_PANELS_IN_SERIES} panels/string.",
            )
        with col_b:
            parallel_strings = st.number_input(
                "Strings wired in parallel",
                min_value=1, value=1, step=1,
                help="1 = single-string site. More than 1 automatically uses "
                     "the multi-string model, normalized to your actual count.",
            )
        string_config = resolve_string_config(parallel_strings)
        st.caption(
            f"Using the **{string_config}** model "
            f"(trained on a {REFERENCE_PARALLEL_STRINGS[string_config]}-parallel-string reference)."
        )
        user_panel_specs = panel_spec_inputs("upload")

    with st.spinner("Running diagnostic model..."):
        try:
            if route == "image":
                prediction = run_cnn_inference(uploaded_file)
            elif route == "rf":
                prediction = run_rf_inference(
                    uploaded_file, string_config, panels_in_series, parallel_strings, user_panel_specs
                )
            elif route == "lstm":
                prediction = run_lstm_inference(uploaded_file)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    show_report(prediction)


def main():
    st.set_page_config(page_title="Solar PV Diagnostic Assistant", page_icon="☀️", layout="centered")
    inject_theme()
    render_hero()

    tab1, tab2 = st.tabs(["📁  Upload File", "⌨️  Manual Entry"])
    with tab1:
        upload_tab()
    with tab2:
        manual_entry_tab()


if __name__ == "__main__":
    main()
