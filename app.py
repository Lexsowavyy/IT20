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
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp                       { background-color: #0f1923; color: #e8dcc8; }
[data-testid="stSidebar"]    { background-color: #162230; border-right: 1px solid #2a3a4a; }
.hero-title   { font-family:'Playfair Display',serif; font-size:2.4rem; font-weight:700; color:#e8dcc8; line-height:1.2; margin-bottom:0.2rem; }
.hero-sub     { font-size:0.95rem; color:#8a9db5; font-weight:300; margin-bottom:1rem; }
.section-label{ font-size:0.68rem; font-weight:500; letter-spacing:0.18em; text-transform:uppercase; color:#c9a96e; margin-bottom:0.5rem; margin-top:1.2rem; }
.card         { background:#162230; border:1px solid #2a3a4a; border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
.result-cancel   { background:linear-gradient(135deg,#3d1a1a,#2a1010); border:1px solid #c0392b; border-left:4px solid #e74c3c; border-radius:12px; padding:1.6rem 2rem; text-align:center; }
.result-no-cancel{ background:linear-gradient(135deg,#0d2b1a,#0a1f14); border:1px solid #1e8449; border-left:4px solid #27ae60; border-radius:12px; padding:1.6rem 2rem; text-align:center; }
.result-prob  { font-size:3rem; font-weight:700; font-family:'Playfair Display',serif; }
.result-label { font-size:0.8rem; color:#8a9db5; letter-spacing:0.1em; text-transform:uppercase; }
.metric-row   { display:flex; gap:1rem; margin-top:0.8rem; }
.metric-box   { flex:1; background:#1c2e3e; border:1px solid #2a3a4a; border-radius:8px; padding:0.8rem; text-align:center; }
.metric-val   { font-size:1.3rem; font-weight:700; color:#c9a96e; font-family:'Playfair Display',serif; }
.metric-lbl   { font-size:0.68rem; color:#8a9db5; text-transform:uppercase; letter-spacing:0.1em; }
.stButton > button { background:linear-gradient(135deg,#c9a96e,#a07840); color:#0f1923; border:none; border-radius:8px; font-family:'DM Sans',sans-serif; font-weight:500; font-size:1rem; width:100%; padding:0.65rem; }
.stButton > button:hover { opacity:0.88; color:#0f1923; }
.stSelectbox > div > div, .stNumberInput > div > div > input { background-color:#1c2e3e !important; border:1px solid #2a3a4a !important; color:#e8dcc8 !important; border-radius:8px !important; }
.stTabs [data-baseweb="tab-list"] { background-color:#162230; border-radius:8px; }
.stTabs [data-baseweb="tab"]      { color:#8a9db5; }
.stTabs [aria-selected="true"]    { color:#c9a96e !important; }
.gold-divider { border:none; border-top:1px solid #c9a96e33; margin:1rem 0; }
label { color:#b0bec5 !important; font-size:0.85rem !important; }
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1rem;'>
        <div style='font-family:Playfair Display,serif;font-size:1.3rem;color:#c9a96e;font-weight:700;'>🏨 HotelPredict</div>
        <div style='font-size:0.78rem;color:#8a9db5;margin-top:0.2rem;'>Cancellation Risk Engine</div>
    </div>
    <hr style='border-color:#2a3a4a;margin-bottom:1rem;'>
    """, unsafe_allow_html=True)

    status_color = "#27ae60" if model_ok else "#e74c3c"
    status_text  = "✓ Loaded" if model_ok else "✗ Error"

    st.markdown("<div class='section-label'>Model Info</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='card' style='padding:0.9rem 1rem;'>
        <div style='font-size:0.8rem;color:#8a9db5;'>Algorithm</div>
        <div style='color:#e8dcc8;font-weight:500;'>Random Forest (Tuned)</div>
        <div style='font-size:0.8rem;color:#8a9db5;margin-top:0.5rem;'>Features</div>
        <div style='color:#c9a96e;font-weight:600;'>13 input variables</div>
        <div style='font-size:0.8rem;color:#8a9db5;margin-top:0.5rem;'>Test Acc / AUC</div>
        <div style='color:#c9a96e;font-weight:600;'>82.8% &nbsp;|&nbsp; 0.9031</div>
        <div style='font-size:0.8rem;color:#8a9db5;margin-top:0.5rem;'>F1-Score</div>
        <div style='color:#c9a96e;font-weight:600;'>71.7%</div>
        <div style='font-size:0.8rem;color:#8a9db5;margin-top:0.5rem;'>Status</div>
        <div style='color:{status_color};font-weight:500;'>{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Session Stats</div>", unsafe_allow_html=True)
    stats = fetch_stats()
    total, n_canceled, avg_prob, avg_adr = stats if stats and stats[0] else (0, 0, 0.0, 0.0)
    st.markdown(f"""
    <div class='card' style='padding:0.9rem 1rem;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem;'>
            <span style='font-size:0.8rem;color:#8a9db5;'>Total Predictions</span>
            <span style='color:#c9a96e;font-weight:600;'>{total or 0}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem;'>
            <span style='font-size:0.8rem;color:#8a9db5;'>Cancellations Flagged</span>
            <span style='color:#e74c3c;font-weight:600;'>{n_canceled or 0}</span>
        </div>
        <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem;'>
            <span style='font-size:0.8rem;color:#8a9db5;'>Avg Cancel Prob</span>
            <span style='color:#c9a96e;font-weight:600;'>{avg_prob or 0}%</span>
        </div>
        <div style='display:flex;justify-content:space-between;'>
            <span style='font-size:0.8rem;color:#8a9db5;'>Avg ADR</span>
            <span style='color:#c9a96e;font-weight:600;'>€{avg_adr or 0}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Rate Reference</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='padding:0.9rem 1rem;font-size:0.75rem;color:#8a9db5;line-height:1.8;'>
        All cancel rates are pre-computed from training data.<br>
        🔴 High &gt;35% &nbsp;|&nbsp; 🟡 Mid 25–35% &nbsp;|&nbsp; 🟢 Low &lt;25%
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Booking Cancellation Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Assess cancellation risk for hotel reservations using a trained Random Forest model.</div>", unsafe_allow_html=True)
st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)

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
            hotel        = st.selectbox("Hotel Type",   HOTEL_OPTIONS)
            deposit_type = st.selectbox("Deposit Type", DEPOSIT_OPTIONS)
        with c2:
            lead_time       = st.number_input("Lead Time (days)", 0, 737, 30)
            booking_changes = st.number_input("Booking Changes",  0, 20,  0)
        with c3:
            adr     = st.number_input("ADR (€)",        0.0, 510.0, 100.0, 5.0)
            parking = st.number_input("Parking Spaces", 0,   8,     0)

        # ── Guest Origin ──────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Guest Origin & Booking Channel</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            country     = st.selectbox(
                "Country of Origin",
                COUNTRY_OPTIONS,
                index=COUNTRY_OPTIONS.index("PRT"),
                help="Select guest country. Cancel rate is pre-computed from training data."
            )
            agency_type = st.selectbox(
                "Booking Channel",
                AGENCY_OPTIONS,
                help="How the booking was made. Cancel rate is pre-computed from training data."
            )
        with c2:
            country_rate = COUNTRY_CANCEL_RATES.get(country, GLOBAL_MEAN)
            agency_rate  = AGENCY_RATES[agency_type]
            c_color = "#e74c3c" if country_rate > 0.35 else "#c9a96e" if country_rate > 0.25 else "#27ae60"
            a_color = "#e74c3c" if agency_rate  > 0.35 else "#c9a96e" if agency_rate  > 0.25 else "#27ae60"
            st.markdown(f"""
            <div style='display:flex;flex-direction:column;gap:0.6rem;margin-top:0.3rem;'>
                <div style='background:#1c2e3e;border:1px solid #2a3a4a;border-radius:8px;padding:0.6rem 1rem;'>
                    <div style='font-size:0.7rem;color:#8a9db5;text-transform:uppercase;letter-spacing:0.1em;'>
                        Country Cancel Rate
                    </div>
                    <div style='font-size:1.3rem;font-weight:700;color:{c_color};font-family:Playfair Display,serif;'>
                        {country_rate*100:.1f}%
                    </div>
                </div>
                <div style='background:#1c2e3e;border:1px solid #2a3a4a;border-radius:8px;padding:0.6rem 1rem;'>
                    <div style='font-size:0.7rem;color:#8a9db5;text-transform:uppercase;letter-spacing:0.1em;'>
                        Channel Cancel Rate
                    </div>
                    <div style='font-size:1.3rem;font-weight:700;color:{a_color};font-family:Playfair Display,serif;'>
                        {agency_rate*100:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Stay & Guests ─────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Stay & Guests</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            total_nights = st.number_input("Total Nights",      0, 60, 3)
            total_guests = st.number_input("Total Guests",      1, 10, 2)
        with c2:
            special_req   = st.number_input("Special Requests",  0, 5,  0)
            got_requested = st.selectbox("Got Requested Room?",  ROOM_OPTIONS)
        with c3:
            is_repeated = st.selectbox("Repeated Guest?", REPEATED_OPTIONS)

        # ── Guest History ─────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Guest History</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            prev_cancel = st.number_input("Previous Cancellations",   0, 26, 0)
        with c2:
            prev_ok     = st.number_input("Previous Bookings (Kept)", 0, 72, 0)

    # ── Result Panel ──────────────────────────────────────────────────────────
    with col_result:
        st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Cancellation Risk", type="primary")

        if predict_btn:
            if not model_ok:
                st.error(f"Model not loaded: {model_err}")
            else:
                raw = {
                    'lead_time'                     : lead_time,
                    'adr'                           : adr,
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

                cancel_prob, prediction, feature_row = predict(model, scaler, raw)
                stay_prob = 1 - cancel_prob

                # Save to DB
                insert_prediction(
                    created_at     = datetime.now().isoformat(timespec="seconds"),
                    hotel          = hotel,
                    lead_time      = lead_time,
                    deposit_type   = deposit_type,
                    market_segment = f"{country} / {agency_type}",
                    adr            = adr,
                    features_json  = json.dumps(feature_row),
                    prediction     = prediction,
                    cancel_prob    = round(cancel_prob, 4),
                )

                # ── Result display ─────────────────────────────────────────
                if prediction == 1:
                    st.markdown(f"""
                    <div class='result-cancel'>
                        <div class='result-label'>PREDICTION</div>
                        <div style='font-family:Playfair Display,serif;font-size:1.7rem;font-weight:700;color:#e74c3c;'>⚠ Likely to Cancel</div>
                        <div class='result-prob' style='color:#e74c3c;'>{cancel_prob*100:.1f}%</div>
                        <div class='result-label'>cancellation probability</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-no-cancel'>
                        <div class='result-label'>PREDICTION</div>
                        <div style='font-family:Playfair Display,serif;font-size:1.7rem;font-weight:700;color:#27ae60;'>✓ Likely to Stay</div>
                        <div class='result-prob' style='color:#27ae60;'>{stay_prob*100:.1f}%</div>
                        <div class='result-label'>retention probability</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.75rem;color:#8a9db5;margin-bottom:0.3rem;'>Cancellation probability</div>", unsafe_allow_html=True)
                st.progress(cancel_prob)

                st.markdown(f"""
                <div class='metric-row'>
                    <div class='metric-box'>
                        <div class='metric-val'>{cancel_prob*100:.1f}%</div>
                        <div class='metric-lbl'>Cancel Risk</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>{stay_prob*100:.1f}%</div>
                        <div class='metric-lbl'>Retention</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val' style='color:{a_color};'>{agency_rate*100:.1f}%</div>
                        <div class='metric-lbl'>Channel Rate</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Risk badge & advice ────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                if cancel_prob >= 0.75:
                    badge  = "<span style='background:#c0392b22;border:1px solid #c0392b;color:#e74c3c;border-radius:20px;padding:0.2rem 0.8rem;font-size:0.75rem;'>🔴 VERY HIGH RISK</span>"
                    advice = "Require a non-refundable deposit or full prepayment. Set up an overbooking buffer."
                elif cancel_prob >= 0.52:
                    badge  = "<span style='background:#c0392b22;border:1px solid #c0392b;color:#e74c3c;border-radius:20px;padding:0.2rem 0.8rem;font-size:0.75rem;'>🔴 HIGH RISK</span>"
                    advice = "Send a confirmation reminder 1 week before arrival. Consider requesting a partial deposit."
                elif cancel_prob >= 0.30:
                    badge  = "<span style='background:#7d600022;border:1px solid #c9a96e;color:#c9a96e;border-radius:20px;padding:0.2rem 0.8rem;font-size:0.75rem;'>🟡 MODERATE RISK</span>"
                    advice = "Monitor this booking. Follow up if no special requests are added."
                else:
                    badge  = "<span style='background:#1e844922;border:1px solid #1e8449;color:#27ae60;border-radius:20px;padding:0.2rem 0.8rem;font-size:0.75rem;'>🟢 LOW RISK</span>"
                    advice = "Booking looks secure. Standard follow-up recommended."

                st.markdown(f"""
                <div style='margin-top:0.6rem;'>
                    {badge}
                    <div style='font-size:0.8rem;color:#8a9db5;margin-top:0.5rem;line-height:1.6;'>
                        💡 {advice}
                    </div>
                </div>""", unsafe_allow_html=True)

                with st.expander("📋 Feature Summary"):
                    st.dataframe(
                        pd.DataFrame([feature_row]).T.rename(columns={0: "Value"}),
                        use_container_width=True
                    )
        else:
            st.markdown("""
            <div style='background:#162230;border:1px dashed #2a3a4a;border-radius:12px;
                        padding:2.5rem 1.5rem;text-align:center;color:#4a6070;'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>🏨</div>
                <div style='font-size:0.9rem;'>Fill in the booking details<br>
                and click <strong style="color:#c9a96e">Predict</strong>.</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<div class='section-label'>Recent Predictions (Last 30)</div>", unsafe_allow_html=True)

    rows = fetch_latest(30)
    if rows:
        df_hist = pd.DataFrame(rows, columns=[
            "ID", "Timestamp", "Hotel", "Lead Time",
            "Deposit Type", "Country / Channel", "ADR (€)",
            "Prediction", "Cancel Probability"
        ])
        df_hist["Prediction"]         = df_hist["Prediction"].map({1: "⚠ Cancel", 0: "✓ Stay"})
        df_hist["Cancel Probability"] = (df_hist["Cancel Probability"] * 100).round(1).astype(str) + "%"
        df_hist["ADR (€)"]            = df_hist["ADR (€)"].map(lambda x: f"€{x:,.2f}")
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        raw_rows = fetch_latest(30)
        dl_df = pd.DataFrame(raw_rows, columns=[
            "ID", "Timestamp", "Hotel", "Lead Time",
            "Deposit Type", "Country / Channel", "ADR", "Prediction", "Cancel Prob"
        ])
        st.download_button(
            label     = "⬇ Download History CSV",
            data      = dl_df.to_csv(index=False).encode("utf-8"),
            file_name = "hotel_predictions_history.csv",
            mime      = "text/csv"
        )
    else:
        st.markdown("""
        <div style='background:#162230;border:1px dashed #2a3a4a;border-radius:12px;
                    padding:2rem;text-align:center;color:#4a6070;'>
            <div style='font-size:1.5rem;margin-bottom:0.4rem;'>🕓</div>
            <div style='font-size:0.9rem;'>No predictions yet.<br>
            Make a prediction in the <strong style="color:#c9a96e">Predict</strong> tab first.</div>
        </div>""", unsafe_allow_html=True)