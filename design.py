"""
Visual design for the Solar PV Diagnostic Assistant.
Theme tokens: dusk-navy backgrounds, sunrise-amber + PV-teal accents.
Fonts: Barlow Condensed (headers), IBM Plex Sans (body), IBM Plex Mono (data).
Result is rendered as a perforated "work-order ticket".
"""

import html
from datetime import datetime

import streamlit as st

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


def render_html(raw: str):
    """st.markdown treats 4+ space indents as a code block, even inside HTML,
    so strip per-line indentation before rendering f-string HTML."""
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


def _is_normal_reading(detection: str) -> bool:
    d = detection.lower()
    return "normal" in d or "no failure" in d


def show_report(prediction: dict):
    """Renders the prediction dict as a diagnostic ticket, plus a debug
    expander for any '_debug_*' keys."""
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

    if "_debug_class_probabilities" in prediction or "_debug_features_used" in prediction:
        with st.expander("Debug details"):
            if "_debug_features_used" in prediction:
                st.write("**Feature values sent to the model (in order):**")
                st.json(prediction["_debug_features_used"])
            if "_debug_class_probabilities" in prediction:
                st.write("**Class probabilities:**")
                st.json(prediction["_debug_class_probabilities"])


def render_technician_notes(note: str):
    """Renders the agent-generated maintenance note under the ticket."""
    render_html(
        f"""
        <div class="pv-ticket-notes-label">Technician Notes — Agent AI</div>
        <div class="pv-ticket-notes">{html.escape(note)}</div>
        """
    )
