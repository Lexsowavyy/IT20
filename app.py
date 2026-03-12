import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from db import init_db, insert_prediction, fetch_latest, fetch_stats

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HotelPredict — Cancellation Risk Engine",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:      #080e14;
    --bg-mid:       #0d1720;
    --bg-card:      #111d28;
    --bg-elevated:  #162332;
    --border:       #1e3045;
    --border-light: #243a52;
    --gold:         #d4a84b;
    --gold-light:   #e8c577;
    --gold-dim:     #7a5f28;
    --text-primary: #e8e0d0;
    --text-secondary:#8fa8c0;
    --text-muted:   #4a6275;
    --red:          #e05252;
    --red-dim:      #5c1e1e;
    --green:        #4cba7a;
    --green-dim:    #1a4a30;
    --amber:        #e0a03a;
    --amber-dim:    #4a3210;
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg-deep);
    color: var(--text-primary);
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 10px; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1520 0%, #0d1a28 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gold-dim), var(--gold), var(--gold-dim));
}

/* ─── Main content bg ─── */
.stApp { background-color: var(--bg-deep); }
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

/* ─── Hero ─── */
.hero-wrapper {
    position: relative;
    padding: 2.2rem 2.5rem;
    background: linear-gradient(135deg, #0d1a28 0%, #111f2e 60%, #0a1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 1.8rem;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--gold-light), var(--gold), transparent);
}
.hero-tag {
    display: inline-block;
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    background: rgba(212,168,75,0.1);
    border: 1px solid rgba(212,168,75,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.15;
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    font-size: 0.88rem;
    color: var(--text-secondary);
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ─── Section Labels ─── */
.section-label {
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.7rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-light), transparent);
}

/* ─── Cards ─── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--border-light); }

/* ─── Info Card (sidebar) ─── */
.info-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.7rem;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
}
.info-row:last-child { border-bottom: none; padding-bottom: 0; }
.info-key { font-size: 0.92rem; color: var(--text-secondary); }
.info-val { font-size: 1.05rem; font-weight: 600; color: var(--gold); font-family: 'Cormorant Garamond', serif; }

/* ─── Result Panels ─── */
.result-cancel {
    background: linear-gradient(160deg, #1a0a0a 0%, #2a0e0e 50%, #1a0a0a 100%);
    border: 1px solid #8b1a1a;
    border-top: 3px solid var(--red);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-cancel::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(224,82,82,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.result-no-cancel {
    background: linear-gradient(160deg, #081a10 0%, #0e2a18 50%, #081a10 100%);
    border: 1px solid #1a6b38;
    border-top: 3px solid var(--green);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-no-cancel::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(76,186,122,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.result-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin-bottom: 0.4rem;
}
.result-prob {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.result-label {
    font-size: 0.68rem;
    color: var(--text-secondary);
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

/* ─── Metric Boxes ─── */
.metric-row { display: flex; gap: 0.7rem; margin-top: 1rem; }
.metric-box {
    flex: 1;
    background: rgba(13,23,36,0.8);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 0.5rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-box:hover { border-color: var(--border-light); transform: translateY(-1px); }
.metric-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--gold);
    font-family: 'Cormorant Garamond', serif;
    line-height: 1.2;
}
.metric-lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ─── Risk Badge ─── */
.risk-badge-wrap { margin-top: 1rem; }
.badge-very-high {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(192,57,43,0.15); border: 1px solid #c0392b;
    color: var(--red); border-radius: 20px;
    padding: 0.25rem 0.9rem; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em;
}
.badge-high {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(192,57,43,0.12); border: 1px solid #a93226;
    color: #e57373; border-radius: 20px;
    padding: 0.25rem 0.9rem; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em;
}
.badge-moderate {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(201,169,110,0.12); border: 1px solid var(--gold-dim);
    color: var(--gold); border-radius: 20px;
    padding: 0.25rem 0.9rem; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em;
}
.badge-low {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(30,132,73,0.12); border: 1px solid #1e8449;
    color: var(--green); border-radius: 20px;
    padding: 0.25rem 0.9rem; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em;
}
.advice-box {
    margin-top: 0.7rem;
    background: rgba(13,23,36,0.6);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold-dim);
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ─── Rate Display Boxes ─── */
.rate-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
}
.rate-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.2rem;
}
.rate-value {
    font-size: 1.65rem;
    font-weight: 700;
    font-family: 'Cormorant Garamond', serif;
    line-height: 1;
}

/* ─── Placeholder ─── */
.placeholder-box {
    background: var(--bg-card);
    border: 1px dashed var(--border-light);
    border-radius: 14px;
    padding: 3rem 1.5rem;
    text-align: center;
    color: var(--text-muted);
}

/* ─── Button ─── */
.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, #a07840 50%, var(--gold) 100%);
    background-size: 200% 100%;
    color: #080e14 !important;
    border: none;
    border-radius: 10px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    width: 100%;
    padding: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: background-position 0.4s, box-shadow 0.3s, transform 0.15s;
    box-shadow: 0 4px 20px rgba(212,168,75,0.2);
}
.stButton > button:hover {
    background-position: right center;
    box-shadow: 0 6px 28px rgba(212,168,75,0.35);
    transform: translateY(-1px);
    color: #080e14 !important;
}
.stButton > button:active { transform: translateY(0); }

/* ─── Inputs ─── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: var(--bg-elevated) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: var(--gold-dim) !important;
    box-shadow: 0 0 0 2px rgba(212,168,75,0.15) !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 0.25rem;
    gap: 0.25rem;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 7px;
    padding: 0.5rem 1.2rem;
    font-size: 0.85rem;
    font-family: 'Outfit', sans-serif;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    background: var(--bg-elevated) !important;
}

/* ─── Labels ─── */
label { color: var(--text-secondary) !important; font-size: 0.82rem !important; font-weight: 400 !important; }

/* ─── Divider ─── */
.gold-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.5rem 0 1.5rem;
    position: relative;
}
.gold-divider::after {
    content: '◆';
    position: absolute;
    left: 50%; top: -0.55rem;
    transform: translateX(-50%);
    color: var(--gold-dim);
    font-size: 0.6rem;
    background: var(--bg-deep);
    padding: 0 0.4rem;
}

/* ─── Sidebar brand ─── */
.brand-wrap { padding: 0.8rem 0 1.2rem; }
.brand-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.03em;
    line-height: 1;
}
.brand-sub { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.2rem; letter-spacing: 0.1em; text-transform: uppercase; }

/* ─── Stats row ─── */
.stat-pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.92rem;
}
.stat-pill:last-child { border-bottom: none; }
.stat-pill-key { color: var(--text-secondary); }
.stat-pill-val { font-weight: 600; font-family: 'Cormorant Garamond', serif; font-size: 1.15rem; }

/* ─── Progress bar ─── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--gold-dim), var(--gold)) !important;
    border-radius: 10px !important;
}
.stProgress > div > div > div {
    background: var(--bg-elevated) !important;
    border-radius: 10px !important;
    height: 6px !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GLOBAL_MEAN = 0.2749
BEST_THRESH = 0.5201

MODEL_FILE_ID  = "1jC7qsZQyGZYf3cLoP_zFizreeox2tNsI"
SCALER_FILE_ID = "1eD2cW46mu2ag8XD57y5Pvmehjrfi8feF"

ROOT        = Path(__file__).resolve().parent
MODEL_PATH  = ROOT / "models" / "hotel_model.pkl"
SCALER_PATH = ROOT / "models" / "hotel_scaler.pkl"

FEATURE_NAMES = [
    'lead_time', 'adr', 'total_of_special_requests',
    'required_car_parking_spaces', 'booking_changes',
    'is_repeated_guest', 'total_nights', 'total_guests',
    'got_requested_room', 'cancellation_rate',
    'lead_x_no_deposit', 'agent_cancel_rate', 'country_cancel_rate'
]

COUNTRY_CANCEL_RATES = {
    "AGO": 0.5461, "ARG": 0.2699, "AUS": 0.2436, "AUT": 0.1860,
    "BEL": 0.1990, "BRA": 0.3646, "CHE": 0.2470, "CHN": 0.4874,
    "CN":  0.1954, "CZE": 0.2647, "DEU": 0.1942, "DNK": 0.2399,
    "ESP": 0.2610, "FIN": 0.1479, "FRA": 0.1941, "GBR": 0.1930,
    "HUN": 0.3354, "IND": 0.2636, "IRL": 0.2254, "ISR": 0.3344,
    "ITA": 0.3432, "JPN": 0.1333, "LUX": 0.4009, "MAR": 0.4388,
    "NLD": 0.1871, "NOR": 0.2703, "POL": 0.2152, "PRT": 0.3501,
    "ROU": 0.2447, "RUS": 0.4089, "SWE": 0.2227, "TUR": 0.4318,
    "USA": 0.2482, "Other": 0.2749,
}

AGENCY_RATES = {
    "Direct / Corporate":        0.1534,
    "Corporate":                 0.1873,
    "Aviation":                  0.2194,
    "Offline Travel Agent":      0.3432,
    "Online Travel Agent (OTA)": 0.3672,
    "Group Booking":             0.6106,
}

DEPOSIT_OPTIONS  = ["No Deposit", "Non Refund", "Refundable"]
HOTEL_OPTIONS    = ["City Hotel", "Resort Hotel"]
REPEATED_OPTIONS = ["No", "Yes"]
ROOM_OPTIONS     = ["Yes", "No"]
COUNTRY_OPTIONS  = sorted(COUNTRY_CANCEL_RATES.keys())
AGENCY_OPTIONS   = list(AGENCY_RATES.keys())

# ── ADR in PHP (approx. 1 EUR ≈ 62 PHP) ──────────────────────────────────────
ADR_CURRENCY = "₱"
ADR_LABEL    = "ADR (₱)"
ADR_MAX      = 31620.0   # ≈ €510
ADR_STEP     = 100.0


# ── Download helpers ──────────────────────────────────────────────────────────
def download_if_needed(file_id: str, dest: Path):
    os.makedirs(dest.parent, exist_ok=True)
    if not dest.exists():
        with st.spinner(f"📥 Downloading {dest.name} from Google Drive..."):
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                str(dest), quiet=False
            )


@st.cache_resource
def load_artifacts():
    download_if_needed(MODEL_FILE_ID,  MODEL_PATH)
    download_if_needed(SCALER_FILE_ID, SCALER_PATH)
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ── Prediction helper (from Code 1) ──────────────────────────────────────────
def predict(model, scaler, raw: dict):
    total_past        = raw['previous_cancellations'] + raw['previous_bookings_not_canceled']
    cancellation_rate = raw['previous_cancellations'] / (total_past + 1)
    lead_x_no_deposit = raw['lead_time'] * (1 if raw['deposit_type'] == 'No Deposit' else 0)

    row = {
        'lead_time'                  : raw['lead_time'],
        'adr'                        : raw['adr'],
        'total_of_special_requests'  : raw['total_of_special_requests'],
        'required_car_parking_spaces': raw['required_car_parking_spaces'],
        'booking_changes'            : raw['booking_changes'],
        'is_repeated_guest'          : raw['is_repeated_guest'],
        'total_nights'               : raw['total_nights'],
        'total_guests'               : raw['total_guests'],
        'got_requested_room'         : raw['got_requested_room'],
        'cancellation_rate'          : cancellation_rate,
        'lead_x_no_deposit'          : lead_x_no_deposit,
        'agent_cancel_rate'          : raw['agent_cancel_rate'],
        'country_cancel_rate'        : raw['country_cancel_rate'],
    }

    df        = pd.DataFrame([row])[FEATURE_NAMES]
    df_scaled = scaler.transform(df)
    prob      = float(model.predict_proba(df_scaled)[0][1])
    pred      = int(prob >= BEST_THRESH)
    return prob, pred, row


# ── Init ──────────────────────────────────────────────────────────────────────
init_db()

try:
    model, scaler = load_artifacts()
    model_ok = True
except Exception as e:
    model_ok  = False
    model_err = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    status_color = "#4cba7a" if model_ok else "#e05252"
    status_text  = "✓ Loaded" if model_ok else "✗ Error"

    st.markdown(f"""
    <div class='brand-wrap'>
        <div class='brand-name'>🏨 HotelPredict</div>
        <div class='brand-sub'>Cancellation Risk Engine</div>
    </div>
    <hr style='border:none;border-top:1px solid #1e3045;margin-bottom:1.2rem;'>
    <div class='section-label'>Model Info</div>
    <div class='info-card'>
        <div class='info-row'>
            <span class='info-key'>Algorithm</span>
            <span class='info-val'>Random Forest</span>
        </div>
        <div class='info-row'>
            <span class='info-key'>Features</span>
            <span class='info-val'>13 variables</span>
        </div>
        <div class='info-row'>
            <span class='info-key'>Test Accuracy</span>
            <span class='info-val'>82.8%</span>
        </div>
        <div class='info-row'>
            <span class='info-key'>AUC Score</span>
            <span class='info-val' style='font-size:1.15rem;'>0.9031</span>
        </div>
        <div class='info-row'>
            <span class='info-key'>F1-Score</span>
            <span class='info-val'>71.7%</span>
        </div>
        <div class='info-row'>
            <span class='info-key'>Status</span>
            <span style='color:{status_color};font-weight:600;font-size:0.85rem;'>{status_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Session Stats</div>", unsafe_allow_html=True)
    stats = fetch_stats()
    total, n_canceled, avg_prob, avg_adr = stats if stats and stats[0] else (0, 0, 0.0, 0.0)
    avg_adr_php = round((avg_adr or 0) * 62, 2)

    st.markdown(f"""
    <div class='info-card'>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Total Predictions</span>
            <span class='stat-pill-val' style='color:#d4a84b;'>{total or 0}</span>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Cancellations Flagged</span>
            <span class='stat-pill-val' style='color:#e05252;'>{n_canceled or 0}</span>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Avg Cancel Prob</span>
            <span class='stat-pill-val' style='color:#d4a84b;'>{avg_prob or 0}%</span>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Avg ADR</span>
            <span class='stat-pill-val' style='color:#d4a84b;'>₱{avg_adr_php:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Risk Reference</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card' style='font-size:0.75rem;line-height:2;'>
        <div>All cancel rates are pre-computed<br>from training data.</div>
        <div style='margin-top:0.5rem;display:flex;flex-direction:column;gap:0.3rem;'>
            <span>🔴 <strong style='color:#e05252;'>High</strong> &nbsp; &gt;35%</span>
            <span>🟡 <strong style='color:#d4a84b;'>Mid</strong> &nbsp; &nbsp;25–35%</span>
            <span>🟢 <strong style='color:#4cba7a;'>Low</strong> &nbsp; &nbsp;&lt;25%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-wrapper'>
    <div class='hero-tag'>AI-Powered · Random Forest · Real-Time</div>
    <div class='hero-title'>Booking <span>Cancellation</span> Predictor</div>
    <div class='hero-sub'>Assess cancellation risk for hotel reservations using a trained Random Forest model. Supports revenue optimization and proactive guest management.</div>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_history = st.tabs(["🔍  Predict", "🕓  History"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:

        # ── Booking Information ───────────────────────────────────────────────
        st.markdown("<div class='section-label'>Booking Information</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            hotel        = st.selectbox("Hotel Type",   HOTEL_OPTIONS,  index=None, placeholder="Select hotel type...")
            deposit_type = st.selectbox("Deposit Type", DEPOSIT_OPTIONS, index=None, placeholder="Select deposit type...")
        with c2:
            lead_time       = st.number_input("Lead Time (days)", min_value=0, max_value=737, value=None, step=1, placeholder="Enter days...")
            booking_changes = st.number_input("Booking Changes",  min_value=0, max_value=20,  value=None, step=1, placeholder="Enter number...")
        with c3:
            if "adr_input" not in st.session_state:
                st.session_state["adr_input"] = ""

            def _fmt_adr():
                raw_val = st.session_state["adr_input"].replace(",", "").strip()
                if raw_val:
                    try:
                        val = float(raw_val)
                        st.session_state["adr_input"] = f"{val:,.0f}" if val >= 1000 else f"{val:.0f}"
                    except ValueError:
                        pass

            adr_raw = st.text_input(
                ADR_LABEL,
                key="adr_input",
                on_change=_fmt_adr,
                help="Average Daily Rate in Philippine Peso (₱)"
            )
            adr_php = None
            adr_display_error = False
            if adr_raw.strip():
                try:
                    adr_php = float(adr_raw.replace(",", "").strip())
                    if adr_php < 0 or adr_php > ADR_MAX:
                        adr_display_error = True
                        adr_php = None
                except ValueError:
                    adr_display_error = True
                    adr_php = None
            if adr_display_error:
                st.markdown(
                    "<div style='font-size:0.78rem;color:#e05252;margin-top:-0.4rem;'>✗ Enter a valid amount</div>",
                    unsafe_allow_html=True
                )
            parking = st.number_input("Parking Spaces", min_value=0, max_value=8, value=None, step=1, placeholder="Enter number...")

        # Convert PHP → EUR for model (trained on EUR)
        adr_eur = (adr_php / 62.0) if adr_php is not None else None

        # ── Guest Origin ──────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Guest Origin & Booking Channel</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            country = st.selectbox(
                "Country of Origin",
                COUNTRY_OPTIONS,
                index=None,
                placeholder="Select country...",
                help="Select guest country. Cancel rate is pre-computed from training data."
            )
            agency_type = st.selectbox(
                "Booking Channel",
                AGENCY_OPTIONS,
                index=None,
                placeholder="Select booking channel...",
                help="How the booking was made. Cancel rate is pre-computed from training data."
            )
        with c2:
            if country and agency_type:
                country_rate = COUNTRY_CANCEL_RATES.get(country, GLOBAL_MEAN)
                agency_rate  = AGENCY_RATES[agency_type]
                c_color = "#e05252" if country_rate > 0.35 else "#d4a84b" if country_rate > 0.25 else "#4cba7a"
                a_color = "#e05252" if agency_rate  > 0.35 else "#d4a84b" if agency_rate  > 0.25 else "#4cba7a"
                st.markdown(f"""
                <div style='display:flex;flex-direction:column;gap:0.6rem;margin-top:0.2rem;'>
                    <div class='rate-box'>
                        <div class='rate-label'>Country Cancel Rate</div>
                        <div class='rate-value' style='color:{c_color};'>{country_rate*100:.1f}%</div>
                    </div>
                    <div class='rate-box'>
                        <div class='rate-label'>Channel Cancel Rate</div>
                        <div class='rate-value' style='color:{a_color};'>{agency_rate*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif country:
                country_rate = COUNTRY_CANCEL_RATES.get(country, GLOBAL_MEAN)
                c_color = "#e05252" if country_rate > 0.35 else "#d4a84b" if country_rate > 0.25 else "#4cba7a"
                st.markdown(f"""
                <div style='display:flex;flex-direction:column;gap:0.6rem;margin-top:0.2rem;'>
                    <div class='rate-box'>
                        <div class='rate-label'>Country Cancel Rate</div>
                        <div class='rate-value' style='color:{c_color};'>{country_rate*100:.1f}%</div>
                    </div>
                    <div class='rate-box' style='opacity:0.4;'>
                        <div class='rate-label'>Channel Cancel Rate</div>
                        <div class='rate-value' style='color:#4a6275;'>— %</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif agency_type:
                agency_rate = AGENCY_RATES[agency_type]
                a_color = "#e05252" if agency_rate > 0.35 else "#d4a84b" if agency_rate > 0.25 else "#4cba7a"
                st.markdown(f"""
                <div style='display:flex;flex-direction:column;gap:0.6rem;margin-top:0.2rem;'>
                    <div class='rate-box' style='opacity:0.4;'>
                        <div class='rate-label'>Country Cancel Rate</div>
                        <div class='rate-value' style='color:#4a6275;'>— %</div>
                    </div>
                    <div class='rate-box'>
                        <div class='rate-label'>Channel Cancel Rate</div>
                        <div class='rate-value' style='color:{a_color};'>{agency_rate*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='display:flex;flex-direction:column;gap:0.6rem;margin-top:0.2rem;'>
                    <div class='rate-box' style='opacity:0.35;'>
                        <div class='rate-label'>Country Cancel Rate</div>
                        <div class='rate-value' style='color:#4a6275;'>— %</div>
                    </div>
                    <div class='rate-box' style='opacity:0.35;'>
                        <div class='rate-label'>Channel Cancel Rate</div>
                        <div class='rate-value' style='color:#4a6275;'>— %</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Stay & Guests ─────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Stay & Guests</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            total_nights = st.number_input("Total Nights", min_value=0, max_value=60, value=None, step=1, placeholder="Enter nights...")
            total_guests = st.number_input("Total Guests", min_value=1, max_value=10, value=None, step=1, placeholder="Enter guests...")
        with c2:
            special_req   = st.number_input("Special Requests", min_value=0, max_value=5, value=None, step=1, placeholder="Enter number...")
            got_requested = st.selectbox("Got Requested Room?", ROOM_OPTIONS, index=None, placeholder="Select...")
        with c3:
            is_repeated = st.selectbox("Repeated Guest?", REPEATED_OPTIONS, index=None, placeholder="Select...")

        # ── Guest History ─────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Guest History</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            prev_cancel = st.number_input("Previous Cancellations",   min_value=0, max_value=26, value=None, step=1, placeholder="Enter number...")
        with c2:
            prev_ok     = st.number_input("Previous Bookings (Kept)", min_value=0, max_value=72, value=None, step=1, placeholder="Enter number...")

    # ── Result Panel ──────────────────────────────────────────────────────────
    with col_result:
        st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Cancellation Risk", type="primary")

        if predict_btn:
            if not model_ok:
                st.error(f"Model not loaded: {model_err}")
            else:
                # ── Validate required fields (from Code 2) ─────────────────
                missing = []
                if hotel          is None: missing.append("Hotel Type")
                if deposit_type   is None: missing.append("Deposit Type")
                if lead_time      is None: missing.append("Lead Time")
                if booking_changes is None: missing.append("Booking Changes")
                if adr_php        is None: missing.append("ADR (₱)")
                if parking        is None: missing.append("Parking Spaces")
                if country        is None: missing.append("Country of Origin")
                if agency_type    is None: missing.append("Booking Channel")
                if total_nights   is None: missing.append("Total Nights")
                if total_guests   is None: missing.append("Total Guests")
                if special_req    is None: missing.append("Special Requests")
                if got_requested  is None: missing.append("Got Requested Room?")
                if is_repeated    is None: missing.append("Repeated Guest?")
                if prev_cancel    is None: missing.append("Previous Cancellations")
                if prev_ok        is None: missing.append("Previous Bookings (Kept)")

                if missing:
                    st.warning(f"⚠ Please fill in all required fields: **{', '.join(missing)}**")
                else:
                    raw = {
                        'lead_time'                     : lead_time,
                        'adr'                           : adr_eur,          # EUR for model
                        'total_of_special_requests'     : special_req,
                        'required_car_parking_spaces'   : parking,
                        'booking_changes'               : booking_changes,
                        'is_repeated_guest'             : 1 if is_repeated == "Yes" else 0,
                        'total_nights'                  : total_nights,
                        'total_guests'                  : total_guests,
                        'got_requested_room'            : 1 if got_requested == "Yes" else 0,
                        'previous_cancellations'        : prev_cancel,
                        'previous_bookings_not_canceled': prev_ok,
                        'deposit_type'                  : deposit_type,
                        'agent_cancel_rate'             : AGENCY_RATES[agency_type],
                        'country_cancel_rate'           : COUNTRY_CANCEL_RATES.get(country, GLOBAL_MEAN),
                    }

                    # ── Run prediction (Code 1 logic) ──────────────────────
                    cancel_prob, prediction, feature_row = predict(model, scaler, raw)
                    stay_prob = 1 - cancel_prob

                    # Save to DB (ADR stored in EUR for consistency)
                    insert_prediction(
                        created_at     = datetime.now().isoformat(timespec="seconds"),
                        hotel          = hotel,
                        lead_time      = lead_time,
                        deposit_type   = deposit_type,
                        market_segment = f"{country} / {agency_type}",
                        adr            = round(adr_eur, 2),
                        features_json  = json.dumps(feature_row),
                        prediction     = prediction,
                        cancel_prob    = round(cancel_prob, 4),
                    )

                    # Agency colour for metrics
                    agency_rate_val = AGENCY_RATES[agency_type]
                    a_color_val = (
                        "#e05252" if agency_rate_val > 0.35
                        else "#d4a84b" if agency_rate_val > 0.25
                        else "#4cba7a"
                    )

                    # ── Result display (Code 1 flow, Code 2 styling) ───────
                    if prediction == 1:
                        st.markdown(f"""
                        <div class='result-cancel'>
                            <div class='result-verdict' style='color:#e05252;'>⚠ Likely to Cancel</div>
                            <div class='result-prob' style='color:#e05252;'>{cancel_prob*100:.1f}%</div>
                            <div class='result-label'>cancellation probability</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-no-cancel'>
                            <div class='result-verdict' style='color:#4cba7a;'>✓ Likely to Stay</div>
                            <div class='result-prob' style='color:#4cba7a;'>{stay_prob*100:.1f}%</div>
                            <div class='result-label'>retention probability</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-size:0.72rem;color:#4a6275;margin-bottom:0.4rem;"
                        "letter-spacing:0.08em;text-transform:uppercase;'>Cancellation Probability</div>",
                        unsafe_allow_html=True
                    )
                    st.progress(cancel_prob)

                    st.markdown(f"""
                    <div class='metric-row'>
                        <div class='metric-box'>
                            <div class='metric-val' style='color:#e05252;'>{cancel_prob*100:.1f}%</div>
                            <div class='metric-lbl'>Cancel Risk</div>
                        </div>
                        <div class='metric-box'>
                            <div class='metric-val' style='color:#4cba7a;'>{stay_prob*100:.1f}%</div>
                            <div class='metric-lbl'>Retention</div>
                        </div>
                        <div class='metric-box'>
                            <div class='metric-val' style='color:{a_color_val};'>{agency_rate_val*100:.1f}%</div>
                            <div class='metric-lbl'>Channel Rate</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # ── Risk badge & advice (Code 1 thresholds + labels) ───
                    st.markdown("<br>", unsafe_allow_html=True)
                    if cancel_prob >= 0.75:
                        badge        = "<span class='badge-very-high'>🔴 VERY HIGH RISK</span>"
                        advice       = "Require a non-refundable deposit or full prepayment. Set up an overbooking buffer."
                        border_color = "#c0392b"
                    elif cancel_prob >= 0.52:
                        badge        = "<span class='badge-high'>🔴 HIGH RISK</span>"
                        advice       = "Send a confirmation reminder 1 week before arrival. Consider requesting a partial deposit."
                        border_color = "#a93226"
                    elif cancel_prob >= 0.30:
                        badge        = "<span class='badge-moderate'>🟡 MODERATE RISK</span>"
                        advice       = "Monitor this booking. Follow up if no special requests are added."
                        border_color = "#7a5f28"
                    else:
                        badge        = "<span class='badge-low'>🟢 LOW RISK</span>"
                        advice       = "Booking looks secure. Standard follow-up recommended."
                        border_color = "#1e6b38"

                    st.markdown(f"""
                    <div class='risk-badge-wrap'>
                        {badge}
                        <div class='advice-box' style='border-left-color:{border_color};'>
                            💡 {advice}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    with st.expander("📋 Feature Summary"):
                        display_row = dict(feature_row)
                        display_row['adr (₱)'] = round(adr_php, 2)
                        display_row.pop('adr', None)
                        st.dataframe(
                            pd.DataFrame([display_row]).T.rename(columns={0: "Value"}),
                            use_container_width=True
                        )

        else:
            st.markdown("""
            <div class='placeholder-box'>
                <div style='font-size:0.9rem;color:#4a6275;line-height:1.7;'>
                    Fill in the booking details<br>
                    and click <strong style="color:#d4a84b;">Predict</strong> to assess risk.
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='section-label'>Prediction History</div>", unsafe_allow_html=True)

    rows = fetch_latest(500)

    if rows:
        df_all = pd.DataFrame(rows, columns=[
            "ID", "Timestamp", "Hotel", "Lead Time",
            "Deposit Type", "Country / Channel", "ADR (EUR)",
            "Prediction", "Cancel Probability"
        ])
        df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], errors="coerce")
        df_all["Date"]      = df_all["Timestamp"].dt.date

        today    = datetime.now().date()
        min_date = df_all["Date"].min()
        max_date = df_all["Date"].max()

        # ── Session state for filter ─────────────────────────────────────────
        if "hist_preset"       not in st.session_state: st.session_state["hist_preset"]       = "Today"
        if "hist_date_from"    not in st.session_state: st.session_state["hist_date_from"]    = min_date
        if "hist_date_to"      not in st.session_state: st.session_state["hist_date_to"]      = max_date
        if "hist_pred_filter"  not in st.session_state: st.session_state["hist_pred_filter"]  = "All"
        if "hist_hotel_filter" not in st.session_state: st.session_state["hist_hotel_filter"] = "All"

        # ── Clear button CSS ─────────────────────────────────────────────────
        st.markdown("""
        <style>
        div[data-testid="stButton"].clear-btn > button {
            background: transparent !important;
            border: 1px solid #2a3a4a !important;
            color: #8fa8c0 !important;
            font-size: 0.72rem !important;
            padding: 0.3rem 0.7rem !important;
            border-radius: 6px !important;
            width: auto !important;
            box-shadow: none !important;
            letter-spacing: 0.05em;
        }
        div[data-testid="stButton"].clear-btn > button:hover {
            border-color: #e05252 !important;
            color: #e05252 !important;
            transform: none !important;
            box-shadow: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Filter row ───────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([2.5, 2, 2, 0.8])
        with fc1:
            preset = st.selectbox(
                "Date Range",
                ["Today", "This Week", "Last Month", "Custom Range"],
                index=["Today", "This Week", "Last Month", "Custom Range"].index(
                    st.session_state["hist_preset"]
                ),
                key="hist_preset_sel"
            )
            st.session_state["hist_preset"] = preset

        with fc2:
            pred_filter = st.selectbox(
                "Prediction",
                ["All", "✓ Stay", "⚠ Cancel"],
                index=["All", "✓ Stay", "⚠ Cancel"].index(st.session_state["hist_pred_filter"]),
                key="hist_pred_sel"
            )
            st.session_state["hist_pred_filter"] = pred_filter

        with fc3:
            hotel_options = ["All"] + sorted(df_all["Hotel"].dropna().unique().tolist())
            hotel_idx = hotel_options.index(st.session_state["hist_hotel_filter"]) \
                        if st.session_state["hist_hotel_filter"] in hotel_options else 0
            hotel_filter = st.selectbox(
                "Hotel Type",
                hotel_options,
                index=hotel_idx,
                key="hist_hotel_sel"
            )
            st.session_state["hist_hotel_filter"] = hotel_filter

        with fc4:
            st.markdown("<div style='margin-top:1.78rem;'></div>", unsafe_allow_html=True)
            st.markdown('<div data-testid="stButton" class="clear-btn">', unsafe_allow_html=True)
            clear_clicked = st.button("✕ Clear", key="hist_clear")
            st.markdown('</div>', unsafe_allow_html=True)
            if clear_clicked:
                st.session_state["hist_preset"]       = "Today"
                st.session_state["hist_date_from"]    = min_date
                st.session_state["hist_date_to"]      = max_date
                st.session_state["hist_pred_filter"]  = "All"
                st.session_state["hist_hotel_filter"] = "All"
                st.rerun()

        # ── Custom Range pickers ─────────────────────────────────────────────
        if preset == "Custom Range":
            cr1, cr2 = st.columns(2)
            with cr1:
                custom_from = st.date_input("From", value=st.session_state["hist_date_from"],
                                            min_value=min_date, max_value=max_date, key="hist_cf")
                st.session_state["hist_date_from"] = custom_from
            with cr2:
                custom_to = st.date_input("To", value=st.session_state["hist_date_to"],
                                          min_value=min_date, max_value=max_date, key="hist_ct")
                st.session_state["hist_date_to"] = custom_to
            date_from = custom_from
            date_to   = custom_to
        elif preset == "Today":
            date_from = today
            date_to   = today
        elif preset == "This Week":
            date_from = today - pd.Timedelta(days=today.weekday())
            date_to   = today
        elif preset == "Last Month":
            date_from = (today.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
            date_to   = today.replace(day=1) - pd.Timedelta(days=1)
        else:
            date_from = min_date
            date_to   = max_date

        # ── Apply filters ────────────────────────────────────────────────────
        mask        = (df_all["Date"] >= date_from) & (df_all["Date"] <= date_to)
        df_filtered = df_all[mask].copy()
        df_filtered["Prediction_label"] = df_filtered["Prediction"].map({1: "⚠ Cancel", 0: "✓ Stay"})

        if pred_filter != "All":
            df_filtered = df_filtered[df_filtered["Prediction_label"] == pred_filter]
        if hotel_filter != "All":
            df_filtered = df_filtered[df_filtered["Hotel"] == hotel_filter]

        # ── Summary stats ────────────────────────────────────────────────────
        total_f    = len(df_filtered)
        cancel_f   = int((df_filtered["Prediction"] == 1).sum())
        stay_f     = total_f - cancel_f
        avg_prob_f = df_filtered["Cancel Probability"].mean() * 100 if total_f > 0 else 0
        avg_adr_f  = df_filtered["ADR (EUR)"].mean() * 62 if total_f > 0 else 0

        st.markdown(f"""
        <div style='display:flex;gap:0.8rem;margin:0.8rem 0 1rem;'>
            <div class='metric-box' style='flex:1;'>
                <div class='metric-val' style='color:#d4a84b;'>{total_f}</div>
                <div class='metric-lbl'>Total</div>
            </div>
            <div class='metric-box' style='flex:1;'>
                <div class='metric-val' style='color:#4cba7a;'>{stay_f}</div>
                <div class='metric-lbl'>Stay</div>
            </div>
            <div class='metric-box' style='flex:1;'>
                <div class='metric-val' style='color:#e05252;'>{cancel_f}</div>
                <div class='metric-lbl'>Cancel</div>
            </div>
            <div class='metric-box' style='flex:1;'>
                <div class='metric-val' style='color:#d4a84b;'>{avg_prob_f:.1f}%</div>
                <div class='metric-lbl'>Avg Risk</div>
            </div>
            <div class='metric-box' style='flex:1;'>
                <div class='metric-val' style='color:#d4a84b;'>₱{avg_adr_f:,.0f}</div>
                <div class='metric-lbl'>Avg ADR</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if total_f == 0:
            st.markdown("""
            <div class='placeholder-box' style='padding:1.5rem;'>
                <div style='font-size:0.9rem;color:#4a6275;'>No records match the selected filters.</div>
            </div>""", unsafe_allow_html=True)
        else:
            df_display = df_filtered[[
                "ID", "Timestamp", "Hotel", "Lead Time",
                "Deposit Type", "Country / Channel", "ADR (EUR)",
                "Prediction_label", "Cancel Probability"
            ]].copy()
            df_display.rename(columns={"Prediction_label": "Prediction"}, inplace=True)
            df_display["Timestamp"]          = df_display["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            df_display["ADR (₱)"]            = df_display["ADR (EUR)"].apply(lambda x: f"₱{x*62:,.0f}")
            df_display["Cancel Probability"] = (df_display["Cancel Probability"] * 100).round(1).astype(str) + "%"
            df_display.drop(columns=["ADR (EUR)"], inplace=True)

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            dl_df = df_filtered[[
                "ID", "Timestamp", "Hotel", "Lead Time",
                "Deposit Type", "Country / Channel", "ADR (EUR)",
                "Prediction_label", "Cancel Probability"
            ]].copy()
            dl_df.rename(columns={"Prediction_label": "Prediction"}, inplace=True)
            dl_df["ADR (₱)"]            = dl_df["ADR (EUR)"].apply(lambda x: round(x * 62, 2))
            dl_df["Cancel Probability"] = (dl_df["Cancel Probability"] * 100).round(1)
            dl_df["Timestamp"]          = dl_df["Timestamp"].astype(str)
            dl_df.drop(columns=["ADR (EUR)"], inplace=True)

            st.download_button(
                label     = "⬇ Download Filtered CSV",
                data      = dl_df.to_csv(index=False).encode("utf-8"),
                file_name = f"hotel_predictions_{date_from}_to_{date_to}.csv",
                mime      = "text/csv"
            )
    else:
        st.markdown("""
        <div class='placeholder-box'>
            <div style='font-size:0.9rem;color:#4a6275;line-height:1.7;'>
                No predictions yet.<br>
                Make a prediction in the <strong style="color:#d4a84b;">Predict</strong> tab first.
            </div>
        </div>""", unsafe_allow_html=True)