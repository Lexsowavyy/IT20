import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import gdown
import joblib
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
.stNumberInput > div > div > input,
.stSlider > div { 
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
.brand-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ─── Stats pill ─── */
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

MODEL_FILE_ID  = "18MmdDol6tbTwAQc_WK16Xbi_yrd_1ndh"
SCALER_FILE_ID = "1CxgQJ1jZEuMbDkxEH9Z6TAISbKnOEFMs"

ROOT        = Path(__file__).resolve().parent
MODEL_PATH  = ROOT / "models" / "hotel_model.pkl"
SCALER_PATH = ROOT / "models" / "hotel_scaler(1).pkl"

# ── FEATURE_NAMES: must match EXACTLY what the .pkl was trained on ───────────
# Source: db.py header comment — 19 model features
FEATURE_NAMES = [
    'agent_cancel_rate',
    'same_room',
    'country_cancel_rate',
    'lead_time',
    'cancellation_rate',
    'lead_x_no_deposit',
    'adr',
    'total_guests',
    'arrival_date_year',
    'total_nights',
    'stays_in_week_nights',
    'adults',
    'children',
    'stays_in_weekend_nights',
    'is_family',
    'previous_cancellations',
    'adr_per_night',
    'arrival_date_day_of_month',
    'days_in_waiting_list',
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


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(model, scaler, raw: dict):
    # Derived features
    total_past        = raw['previous_cancellations'] + raw['previous_bookings_not_canceled']
    cancellation_rate = raw['previous_cancellations'] / (total_past + 1)
    lead_x_no_deposit = raw['lead_time'] * (1 if raw['deposit_type'] == 'No Deposit' else 0)
    adr_per_night     = raw['adr'] / max(raw['total_nights'], 1)
    total_guests      = raw['adults'] + raw['children']
    is_family         = int(raw['children'] > 0 or raw['babies'] > 0)

    # Build all feature values — order does not matter here, we reorder below
    row = {
        'agent_cancel_rate'         : raw['agent_cancel_rate'],
        'same_room'                 : raw['got_requested_room'],
        'country_cancel_rate'       : raw['country_cancel_rate'],
        'lead_time'                 : raw['lead_time'],
        'cancellation_rate'         : cancellation_rate,
        'lead_x_no_deposit'         : lead_x_no_deposit,
        'adr'                       : raw['adr'],
        'total_guests'              : total_guests,
        'arrival_date_year'         : raw['arrival_date_year'],
        'total_nights'              : raw['total_nights'],
        'stays_in_week_nights'      : raw['stays_in_week_nights'],
        'adults'                    : raw['adults'],
        'children'                  : raw['children'],
        'stays_in_weekend_nights'   : raw['stays_in_weekend_nights'],
        'is_family'                 : is_family,
        'previous_cancellations'    : raw['previous_cancellations'],
        'adr_per_night'             : adr_per_night,
        'arrival_date_day_of_month' : raw['arrival_date_day_of_month'],
        'days_in_waiting_list'      : raw['days_in_waiting_list'],
    }

    # ── Step 1: scale using scaler's exact 19-feature order ─────────────────
    scaler_cols = list(scaler.feature_names_in_)
    df_for_scaler = pd.DataFrame([row])[scaler_cols]
    df_scaled_arr = scaler.transform(df_for_scaler)

    # ── Step 2: pass only the N features the RF model was trained on ────────
    # Older sklearn models don't store feature_names_in_, but n_features_in_
    # (the count) is always available. We take the first N scaled columns,
    # which matches the order the model was fitted on.
    n_model_feats  = model.n_features_in_
    df_scaled_full = pd.DataFrame(df_scaled_arr, columns=scaler_cols)
    df_for_model   = df_scaled_full.iloc[:, :n_model_feats]

    prob = float(model.predict_proba(df_for_model)[0][1])
    pred = int(prob >= BEST_THRESH)
    return prob, pred, row


# ── Init DB ───────────────────────────────────────────────────────────────────
try:
    init_db()
    db_ok = True
except Exception as e:
    db_ok = False
    st.warning(f"Database init failed: {e}")

# ── Load model ────────────────────────────────────────────────────────────────
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
    status_text  = "✓ Loaded"  if model_ok else "✗ Error"

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
            <span class='info-val'>{len(FEATURE_NAMES)} variables</span>
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
    try:
        stats = fetch_stats()
        if stats and stats[0]:
            total      = stats[0] or 0
            n_canceled = stats[1] or 0
            avg_prob   = stats[2] or 0.0
            avg_adr    = stats[3] or 0.0
        else:
            total = n_canceled = avg_prob = avg_adr = 0
    except Exception:
        total = n_canceled = avg_prob = avg_adr = 0

    st.markdown(f"""
    <div class='info-card'>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Total Predictions</span>
            <span class='stat-pill-val' style='color:#d4a84b;'>{total}</span>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Cancellations Flagged</span>
            <span class='stat-pill-val' style='color:#e05252;'>{n_canceled}</span>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Avg Cancel Prob</span>
            <span class='stat-pill-val' style='color:#d4a84b;'>{avg_prob}%</span>
        </div>
        <div class='stat-pill'>
            <span class='stat-pill-key'>Avg ADR</span>
            <span class='stat-pill-val' style='color:#d4a84b;'>${avg_adr:,.2f}</span>
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
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-wrapper'>
    <div class='hero-tag'>AI-Powered · Random Forest · Real-Time</div>
    <div class='hero-title'>Booking <span>Cancellation</span> Predictor</div>
    <div class='hero-sub'>Assess cancellation risk for hotel reservations using a trained Random Forest model.
    Supports revenue optimization and proactive guest management.</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍  Predict", "🕓  History / Stats"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:

        # ── Booking Details ───────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Booking Details</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            lead_time       = st.number_input("Lead Time (days)",    min_value=0,   max_value=1000, value=60)
            arrival_year    = st.selectbox("Arrival Year",           [2015, 2016, 2017], index=2)
        with c2:
            adr             = st.number_input("ADR ($)",             min_value=0.0, max_value=5000.0, value=100.0)
            arrival_day     = st.number_input("Arrival Day of Month",min_value=1,   max_value=31, value=15)
        with c3:
            no_deposit      = st.selectbox("No Deposit?",            ["Yes", "No"])
            days_in_waiting = st.number_input("Days in Waiting List",min_value=0,   max_value=400, value=0)

        # ── Stay Duration ─────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Stay Duration</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            week_nights    = st.number_input("Week Nights",    min_value=0, max_value=50, value=3)
        with c2:
            weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=20, value=1)

        # Derived
        total_nights  = week_nights + weekend_nights
        deposit_type  = "No Deposit" if no_deposit == "Yes" else "Non Refund"

        # ── Guest Composition ─────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Guest Composition</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            adults   = st.number_input("Adults",   min_value=1, max_value=10, value=2)
        with c2:
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        with c3:
            babies   = st.number_input("Babies",   min_value=0, max_value=10, value=0)

        # Derived
        total_guests = adults + children + babies
        is_family    = int(children > 0 or babies > 0)

        # ── Guest History ─────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Guest History</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            got_requested_room         = st.selectbox("Got Requested Room?", ROOM_OPTIONS)
        with c2:
            previous_cancellations     = st.number_input("Previous Cancellations",  min_value=0, max_value=50, value=0)
        with c3:
            previous_bookings_not_canceled = st.number_input("Prev. Bookings Kept", min_value=0, max_value=50, value=0)

        # ── Cancel Rates ──────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Cancellation Rates</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            agent_cancel_rate   = st.number_input(
                f"Agent Cancel Rate  [global = {GLOBAL_MEAN}]",
                min_value=0.0, max_value=1.0,
                value=round(GLOBAL_MEAN, 3), step=0.001, format="%.3f"
            )
            country             = st.selectbox("Country of Origin", COUNTRY_OPTIONS)
            country_cancel_rate = COUNTRY_CANCEL_RATES.get(country, GLOBAL_MEAN)
        with c2:
            c_color = "#e05252" if country_cancel_rate > 0.35 else "#d4a84b" if country_cancel_rate > 0.25 else "#4cba7a"
            a_color = "#e05252" if agent_cancel_rate   > 0.35 else "#d4a84b" if agent_cancel_rate   > 0.25 else "#4cba7a"
            st.markdown(f"""
            <div style='display:flex;flex-direction:column;gap:0.6rem;margin-top:0.2rem;'>
                <div class='rate-box'>
                    <div class='rate-label'>Country Cancel Rate</div>
                    <div class='rate-value' style='color:{c_color};'>{country_cancel_rate*100:.1f}%</div>
                </div>
                <div class='rate-box'>
                    <div class='rate-label'>Agent Cancel Rate</div>
                    <div class='rate-value' style='color:{a_color};'>{agent_cancel_rate*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Result Panel ──────────────────────────────────────────────────────────
    with col_result:
        st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Cancellation Risk", type="primary")

        if predict_btn:
            if not model_ok:
                st.error(f"Model not loaded: {model_err}")
            else:
                raw_input = {
                    # Direct inputs
                    'lead_time'                     : lead_time,
                    'adr'                           : adr,
                    'adults'                        : adults,
                    'children'                      : children,
                    'babies'                        : babies,
                    'stays_in_week_nights'          : week_nights,
                    'stays_in_weekend_nights'       : weekend_nights,
                    'total_nights'                  : total_nights,
                    'arrival_date_year'             : arrival_year,
                    'arrival_date_day_of_month'     : arrival_day,
                    'days_in_waiting_list'          : days_in_waiting,
                    'previous_cancellations'        : previous_cancellations,
                    'previous_bookings_not_canceled': previous_bookings_not_canceled,
                    'got_requested_room'            : 1 if got_requested_room == "Yes" else 0,
                    'deposit_type'                  : deposit_type,
                    'agent_cancel_rate'             : agent_cancel_rate,
                    'country_cancel_rate'           : country_cancel_rate,
                }

                prob, pred, features = predict(model, scaler, raw_input)
                stay_prob = 1 - prob

                # ── Result card ───────────────────────────────────────────────
                if pred == 1:
                    st.markdown(f"""
                    <div class='result-cancel'>
                        <div class='result-verdict' style='color:#e05252;'>⚠ Likely to Cancel</div>
                        <div class='result-prob' style='color:#e05252;'>{prob*100:.1f}%</div>
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
                st.progress(prob)

                st.markdown(f"""
                <div class='metric-row'>
                    <div class='metric-box'>
                        <div class='metric-val' style='color:#e05252;'>{prob*100:.1f}%</div>
                        <div class='metric-lbl'>Cancel Risk</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val' style='color:#4cba7a;'>{stay_prob*100:.1f}%</div>
                        <div class='metric-lbl'>Retention</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val' style='color:{a_color};'>{agent_cancel_rate*100:.1f}%</div>
                        <div class='metric-lbl'>Channel Rate</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Risk badge & advice ───────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if prob >= 0.75:
                    badge        = "<span class='badge-very-high'>🔴 VERY HIGH RISK</span>"
                    advice       = "Require a non-refundable deposit or full prepayment. Set up an overbooking buffer."
                    border_color = "#c0392b"
                elif prob >= 0.52:
                    badge        = "<span class='badge-high'>🔴 HIGH RISK</span>"
                    advice       = "Send a confirmation reminder 1 week before arrival. Consider requesting a partial deposit."
                    border_color = "#a93226"
                elif prob >= 0.30:
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
                    st.dataframe(
                        pd.DataFrame([features]).T.rename(columns={0: "Value"}),
                        use_container_width=True
                    )

                # ── Save to DB ────────────────────────────────────────────────
                if db_ok:
                    try:
                        insert_prediction(
                            created_at                     = datetime.now().isoformat(timespec="seconds"),
                            hotel                          = "City Hotel",
                            market_segment                 = country,
                            deposit_type                   = deposit_type,
                            distribution_channel           = "Direct",
                            customer_type                  = "Transient",
                            lead_time                      = lead_time,
                            adr                            = adr,
                            arrival_date_year              = arrival_year,
                            arrival_date_day_of_month      = arrival_day,
                            days_in_waiting_list           = days_in_waiting,
                            total_of_special_requests      = 0,
                            booking_changes                = 0,
                            is_repeated_guest              = 0,
                            adults                         = adults,
                            children                       = children,
                            babies                         = babies,
                            stays_in_week_nights           = week_nights,
                            stays_in_weekend_nights        = weekend_nights,
                            previous_cancellations         = previous_cancellations,
                            previous_bookings_not_canceled = previous_bookings_not_canceled,
                            total_guests                   = total_guests,
                            is_family                      = is_family,
                            total_nights                   = total_nights,
                            adr_per_night                  = adr / max(total_nights, 1),
                            total_previous_bookings        = previous_cancellations + previous_bookings_not_canceled,
                            cancellation_rate              = features['cancellation_rate'],
                            has_booking_changes            = 0,
                            same_room                      = features['same_room'],
                            request_per_guest              = 0.0,
                            lead_x_no_deposit              = features['lead_x_no_deposit'],
                            agent_cancel_rate              = agent_cancel_rate,
                            country_cancel_rate            = country_cancel_rate,
                            features_json                  = json.dumps(features),
                            prediction                     = pred,
                            cancel_prob                    = round(prob, 4),
                        )
                    except Exception as e:
                        st.warning(f"Failed to save prediction to DB: {e}")

        else:
            st.markdown("""
            <div class='placeholder-box'>
                <div style='font-size:0.9rem;color:#4a6275;line-height:1.7;'>
                    Fill in the booking details<br>
                    and click <strong style="color:#d4a84b;">Predict</strong> to assess risk.
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY / STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── History ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Prediction History</div>", unsafe_allow_html=True)
    try:
        rows = fetch_latest(30)
        if rows:
            df_hist = pd.DataFrame(rows, columns=[
                "ID", "Timestamp", "Hotel", "Lead Time",
                "Deposit Type", "Country / Channel", "ADR",
                "Total Nights", "Total Guests", "Family",
                "Hist. Cancel Rate", "Same Room", "Req/Guest",
                "Prediction", "Cancel Probability",
            ])
            df_hist["Prediction"]         = df_hist["Prediction"].map({1: "⚠ Cancel", 0: "✓ Stay"})
            df_hist["Cancel Probability"] = (df_hist["Cancel Probability"] * 100).round(1).astype(str) + "%"
            df_hist["Family"]             = df_hist["Family"].map({1: "✓ Yes", 0: "No"})
            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            st.download_button(
                label     = "⬇ Download CSV",
                data      = df_hist.to_csv(index=False).encode("utf-8"),
                file_name = f"hotel_predictions_{datetime.now().date()}.csv",
                mime      = "text/csv",
            )
        else:
            st.markdown("""
            <div class='placeholder-box' style='padding:1.5rem;'>
                <div style='font-size:0.9rem;color:#4a6275;'>No prediction history yet.</div>
            </div>""", unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class='placeholder-box' style='padding:1.5rem;'>
            <div style='font-size:0.9rem;color:#4a6275;'>No history available.</div>
        </div>""", unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Session Statistics</div>", unsafe_allow_html=True)
    try:
        stats = fetch_stats()
        if stats and stats[0]:
            t, nc, ap, aa = stats[0], stats[1], stats[2], stats[3]
            cancel_pct = round((nc / t * 100), 1) if t else 0
            st.markdown(f"""
            <div style='display:flex;gap:0.8rem;margin:0.5rem 0 1rem;'>
                <div class='metric-box' style='flex:1;'>
                    <div class='metric-val' style='color:#d4a84b;'>{t}</div>
                    <div class='metric-lbl'>Total</div>
                </div>
                <div class='metric-box' style='flex:1;'>
                    <div class='metric-val' style='color:#4cba7a;'>{t - nc}</div>
                    <div class='metric-lbl'>Stay</div>
                </div>
                <div class='metric-box' style='flex:1;'>
                    <div class='metric-val' style='color:#e05252;'>{nc}</div>
                    <div class='metric-lbl'>Cancel</div>
                </div>
                <div class='metric-box' style='flex:1;'>
                    <div class='metric-val' style='color:#d4a84b;'>{ap}%</div>
                    <div class='metric-lbl'>Avg Risk</div>
                </div>
                <div class='metric-box' style='flex:1;'>
                    <div class='metric-val' style='color:#d4a84b;'>${aa:,.2f}</div>
                    <div class='metric-lbl'>Avg ADR</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='placeholder-box' style='padding:1.5rem;'>
                <div style='font-size:0.9rem;color:#4a6275;'>No stats available yet.</div>
            </div>""", unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div class='placeholder-box' style='padding:1.5rem;'>
            <div style='font-size:0.9rem;color:#4a6275;'>No stats available.</div>
        </div>""", unsafe_allow_html=True)