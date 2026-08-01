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
import html
from datetime import timedelta, datetime

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
        # Workaround for a CrewAI bug: it injects a `cache_breakpoint` marker
        # into messages for Anthropic-style prompt caching, but doesn't strip
        # it for non-Anthropic providers -- Groq's API then rejects the
        # request outright. See https://github.com/crewAIInc/crewAI/issues/5886
        try:
            import crewai.llms.cache as _crewai_cache
            _crewai_cache.mark_cache_breakpoint = lambda msg: msg
        except Exception:
            pass  # crewai internals changed / not present -- safe to ignore
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
# ARRAY-CONFIG SCALING (series panels + parallel strings)
# --------------------------------------------------------------------------
# The RF models were trained on a fixed reference wiring -- 5 panels in
# series per string, and either 1 or 3 strings in parallel (matching which
# of the two trained models is used). Voltage scales with panels in SERIES;
# current scales with strings in PARALLEL; power scales with both
# (Power = Voltage * Current). Temperature/irradiance don't scale with
# either. Mismatching the reference is exactly why a 2-panel or 2-parallel-
# string reading can look like a fault (e.g. OC) to a model trained on a
# different physical size.
REFERENCE_PANELS_IN_SERIES = 5  # panels per string the models were trained on

# How many parallel strings each trained model's data represents.
REFERENCE_PARALLEL_STRINGS = {
    "1-string": 1,
    "3-string": 3,
}

# --------------------------------------------------------------------------
# ARRAY-CONFIG + PANEL-TYPE SCALING
# --------------------------------------------------------------------------
# The RF models were trained on ONE specific simulated panel (see the
# Simulink PV Array block this was built from): 66 cells, Voc 47.42V,
# Isc 15A, Vmp 39.51V, Imp 14.17A -- wired 5 in series, and either 1 or 3
# strings in parallel (matching which of the two trained models is used).
#
# Two independent things can mismatch the training data, and both get
# corrected before scoring:
#   1. ARRAY WIRING -- different series panel count / parallel string count.
#      Voltage scales with SERIES count; current scales with PARALLEL count;
#      power scales with both.
#   2. PANEL TYPE -- a different physical panel model entirely. We convert
#      the user's reading to a fraction of THEIR panel's rated values
#      (per-unit normalization), then re-scale that fraction onto the
#      REFERENCE panel's rated values. This assumes similar cell technology/
#      IV-curve shape between panels; it does not correct for differing
#      temperature coefficients.
# Temperature/irradiance readings are left unchanged by either correction.
REFERENCE_PANELS_IN_SERIES = 5  # panels per string the models were trained on

# How many parallel strings each trained model's data represents.
REFERENCE_PARALLEL_STRINGS = {
    "1-string": 1,
    "3-string": 3,
}

# Nameplate ratings of the exact panel the simulation/training data used
# (from the Simulink PV Array "Block Parameters" dialog).
REFERENCE_PANEL_SPECS = {
    "Voc": 47.42,   # open-circuit voltage (V)
    "Isc": 15.0,    # short-circuit current (A)
    "Vmp": 39.51,   # voltage at max power point (V)
    "Imp": 14.17,   # current at max power point (A)
}
REFERENCE_PANEL_SPECS["Pmax"] = REFERENCE_PANEL_SPECS["Vmp"] * REFERENCE_PANEL_SPECS["Imp"]

# Maps each data column to (a) which array dimension it scales with, and
# (b) which nameplate spec it should be normalized against.
COLUMN_SCALE_CONFIG = {
    "Voc_V":  {"array_dim": "series",   "spec_key": "Voc"},
    "Vmp_V":  {"array_dim": "series",   "spec_key": "Vmp"},
    "Isc_A":  {"array_dim": "parallel", "spec_key": "Isc"},
    "Imp_A":  {"array_dim": "parallel", "spec_key": "Imp"},
    "Pmax_W": {"array_dim": "both",     "spec_key": "Pmax"},
}
# Temp_C, Irr_Wm2 are intentionally absent from this map -- they don't scale
# with array wiring or panel type.


def scale_reading(df: pd.DataFrame, panels_in_series: int, parallel_strings: int,
                   reference_parallel_strings: int, user_panel_specs: dict) -> pd.DataFrame:
    """Project a reading taken from a real array (possibly a different panel
    model, wired differently) up/down to what it would look like on the
    REFERENCE array/panel the model was trained on."""
    if panels_in_series is None or panels_in_series <= 0:
        raise ValueError("Panels in series must be a positive number.")
    if parallel_strings is None or parallel_strings <= 0:
        raise ValueError("Parallel strings must be a positive number.")

    series_ratio = REFERENCE_PANELS_IN_SERIES / panels_in_series
    parallel_ratio = reference_parallel_strings / parallel_strings
    dim_ratio = {"series": series_ratio, "parallel": parallel_ratio,
                 "both": series_ratio * parallel_ratio}

    # Fill in Pmax for the user's panel if not given directly.
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
        panel_ratio = (ref_val / actual_val) if actual_val else 1.0  # no user spec -> assume same panel

        df[col] = df[col] * array_ratio * panel_ratio
    return df

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

def run_rf_inference_from_df(df: pd.DataFrame, string_config: str,
                              panels_in_series: int = REFERENCE_PANELS_IN_SERIES,
                              parallel_strings: int = None,
                              user_panel_specs: dict = None):
    """string_config must be '1-string' or '3-string' (chosen by the user --
    picks which trained model/scaler pair to use). panels_in_series and
    parallel_strings describe the ACTUAL array wiring this reading came from;
    they default to the model's own reference configuration if not given.
    user_panel_specs (dict with Voc/Isc/Vmp/Imp keys) describes the ACTUAL
    panel model in use, from its datasheet; omit/leave None to assume it's
    the same panel the model was trained on."""
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

    # Each scaler remembers the exact feature names/order it was fit on
    # (sklearn stores this as feature_names_in_ when fit on a DataFrame).
    missing = set(scaler.feature_names_in_) - set(df.columns)
    if missing:
        raise ValueError(
            f"Data is missing columns required by the {string_config} model: {sorted(missing)}\n"
            f"Expected columns: {list(scaler.feature_names_in_)}"
        )

    # Normalize the reading to the model's reference array wiring AND
    # reference panel type, so a different panel count, parallel-string
    # count, or physical panel model doesn't look like a fault.
    df = scale_reading(df, panels_in_series, parallel_strings, reference_parallel, user_panel_specs)

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

    panel_note = "reference panel" if not user_panel_specs else "custom panel, normalized"
    result = {
        "data_type": (
            f"Sensor Snapshot ({string_config} model, "
            f"{panels_in_series} panels/string, {parallel_strings} parallel, {panel_note})"
        ),
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
# THEME
# --------------------------------------------------------------------------
# Design tokens:
#   bg / surface / surface-alt : dusk-navy panel backgrounds (pre-dawn sky,
#     not the generic near-black+neon look)
#   accent-amber : sunrise gold, primary accent (buttons, active states)
#   accent-teal  : PV-cell teal, secondary accent (borders, links)
#   success / danger : status colors for Normal vs Fault detections
# Type: Barlow Condensed (industrial display headers) + IBM Plex Sans (body)
#   + IBM Plex Mono (data readouts / the ticket) -- the Plex family has real
#   engineering/electronics heritage, which fits a diagnostics tool.
# Signature element: the diagnostic result renders as a perforated field
#   work-order ticket, echoing the agent's own "maintenance ticket" language.

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #F3F6F9;
    --surface: #FFFFFF;
    --surface-alt: #E9F0F5;
    --line: #D3DEE6;
    --accent-amber: #D9861F;
    --accent-teal: #2E7D74;
    --text-primary: #182530;
    --text-muted: #5C6E7D;
    --success: #2F8F5B;
    --danger: #C24B36;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMain"] .block-container { padding-top: 1.5rem; max-width: 780px; }

/* Hero banner */
.pv-hero {
    border: 1px solid var(--line);
    border-bottom: 3px solid var(--accent-amber);
    background: linear-gradient(180deg, var(--surface-alt) 0%, var(--surface) 100%);
    padding: 1.6rem 1.8rem 1.3rem 1.8rem;
    margin-bottom: 1.6rem;
}
.pv-hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--accent-teal);
    text-transform: uppercase;
}
.pv-hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin: 0.15rem 0 0.35rem 0;
    line-height: 1.05;
}
.pv-hero-sub {
    color: var(--text-muted);
    font-size: 0.92rem;
    max-width: 100%;
}
.pv-hero-horizon {
    width: 100%; height: 28px; margin-top: 1rem; display: block;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-amber) !important;
    border-bottom-color: var(--accent-amber) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--accent-amber) !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--line) !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--accent-teal) !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text-muted) !important; }

/* Buttons */
.stButton>button {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: var(--accent-amber) !important;
    color: #0C1524 !important;
    border: none !important;
    border-radius: 2px !important;
}
.stButton>button:hover { background: #f0b459 !important; }

/* Radio */
[data-testid="stRadio"] label { color: var(--text-primary) !important; }

/* Number inputs / text inputs */
[data-testid="stNumberInput"] input, .stTextInput input {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border-color: var(--line) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-left: 3px solid var(--accent-teal) !important;
    border-radius: 2px !important;
}

/* Alert boxes */
[data-testid="stAlert"] { border-radius: 2px !important; font-family: 'IBM Plex Sans', sans-serif; }

/* Captions */
[data-testid="stCaptionContainer"] { font-family: 'IBM Plex Mono', monospace !important; color: var(--text-muted) !important; }
</style>
"""

HORIZON_SVG = """
<svg class="pv-hero-horizon" viewBox="0 0 600 28" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="24" x2="600" y2="24" stroke="#C3D2DC" stroke-width="1"/>
  <circle cx="40" cy="24" r="9" fill="#D9861F"/>
  <g stroke="#2E7D74" stroke-width="2">
    <line x1="90" y1="24" x2="90" y2="10"/><line x1="102" y1="24" x2="102" y2="6"/>
    <line x1="114" y1="24" x2="114" y2="14"/><line x1="126" y1="24" x2="126" y2="8"/>
    <line x1="138" y1="24" x2="138" y2="12"/><line x1="150" y1="24" x2="150" y2="5"/>
    <line x1="162" y1="24" x2="162" y2="16"/><line x1="174" y1="24" x2="174" y2="9"/>
  </g>
</svg>
"""


def render_html(raw: str):
    """st.markdown treats lines indented 4+ spaces as a Markdown code block,
    even inside an HTML block -- so pretty-indented f-string HTML can get
    partially rendered as literal text. Strip per-line indentation first."""
    lines = raw.strip("\n").split("\n")
    st.markdown("\n".join(line.lstrip() for line in lines), unsafe_allow_html=True)


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_hero():
    render_html(
        f"""
        <div class="pv-hero">
            <div class="pv-hero-title">Solar PV Diagnostic Assistant</div>
            <div class="pv-hero-sub">Upload a panel photo or sensor reading or key one in by hand. The system routes it to the right model and prints a technician ticket.</div>
            {HORIZON_SVG}
        </div>
        """
    )


TICKET_CSS = """
<style>
.pv-ticket {
    background: var(--surface);
    border: 1px solid var(--line);
    margin-top: 0.6rem;
}
.pv-ticket-perf {
    height: 14px;
    background-image: radial-gradient(circle at 9px 7px, var(--bg) 5px, transparent 5.5px);
    background-size: 18px 14px;
    background-repeat: repeat-x;
}
.pv-ticket-body { padding: 0.2rem 1.4rem 1.3rem 1.4rem; }
.pv-ticket-head {
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px dashed var(--line);
    padding-bottom: 0.5rem; margin-bottom: 0.7rem;
}
.pv-ticket-eyebrow {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    font-size: 0.82rem; color: var(--text-muted);
}
.pv-ticket-id {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem; color: var(--text-muted);
}
.pv-ticket-row {
    display: flex; justify-content: space-between; gap: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.92rem; padding: 0.3rem 0;
    border-bottom: 1px dotted var(--line);
}
.pv-ticket-row span:first-child { color: var(--text-muted); }
.pv-ticket-row span:last-child { color: var(--text-primary); font-weight: 500; text-align: right; }
.pv-ticket-status {
    display: inline-block; padding: 0.1rem 0.55rem; border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}
.pv-ticket-status.ok { background: rgba(47,143,91,0.13); color: var(--success); }
.pv-ticket-status.alert { background: rgba(194,75,54,0.13); color: var(--danger); }
.pv-ticket-notes-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    font-size: 0.82rem; color: var(--accent-teal); margin-top: 1rem;
}
.pv-ticket-notes {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem; line-height: 1.5; color: var(--text-primary);
    margin-top: 0.3rem;
}
</style>
"""


def _is_normal_reading(detection: str) -> bool:
    d = detection.lower()
    return "normal" in d or "no failure" in d


def show_report(prediction: dict):
    st.markdown(TICKET_CSS, unsafe_allow_html=True)

    detection = str(prediction["detection"])
    confidence = prediction.get("confidence")
    status_class = "ok" if _is_normal_reading(detection) else "alert"
    status_label = "NOMINAL" if status_class == "ok" else "ATTENTION"
    ticket_id = datetime.now().strftime("PV-%Y%m%d-%H%M%S")

    confidence_row = ""
    if confidence not in (None, "N/A"):
        confidence_row = (
            f'<div class="pv-ticket-row"><span>Confidence</span>'
            f'<span>{html.escape(str(confidence))}%</span></div>'
        )

    render_html(
        f"""
        <div class="pv-ticket">
            <div class="pv-ticket-perf"></div>
            <div class="pv-ticket-body">
                <div class="pv-ticket-head">
                    <span class="pv-ticket-eyebrow">Diagnostic Ticket</span>
                    <span class="pv-ticket-id">{ticket_id}</span>
                </div>
                <div class="pv-ticket-row"><span>Data Processed</span>
                    <span>{html.escape(str(prediction['data_type']))}</span></div>
                <div class="pv-ticket-row"><span>Detection</span>
                    <span>{html.escape(detection)}</span></div>
                {confidence_row}
                <div class="pv-ticket-row"><span>Status</span>
                    <span><span class="pv-ticket-status {status_class}">{status_label}</span></span></div>
            </div>
        </div>
        """
    )

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
            render_html(
                f"""
                <div class="pv-ticket-notes-label">Technician Notes — Agent AI</div>
                <div class="pv-ticket-notes">{html.escape(note)}</div>
                """
            )
        except Exception as e:
            st.warning(
                f"Agent report generation failed ({e}). Showing raw prediction only."
            )


def panel_spec_inputs(key_prefix: str):
    """Renders an expander asking for the user's actual panel datasheet
    specs. Returns a dict (Voc/Isc/Vmp/Imp) or None if left at defaults
    (meaning: assume same panel the model was trained on)."""
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
    if unchanged:
        return None  # unchanged from defaults -- no normalization needed
    return specs


def resolve_string_config(parallel_strings: int) -> str:
    """1 parallel string -> the 1-string model. Anything more than 1 -> the
    multi-string model (trained on a 3-parallel-string reference, but its
    scaling normalizes any actual parallel count onto that reference)."""
    return "1-string" if parallel_strings == 1 else "3-string"


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
    else:
        user_panel_specs = None

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
