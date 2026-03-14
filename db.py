"""
db.py — SQLite persistence layer for Hotel Booking Cancellation Predictor
=========================================================================

19 model features used by the model
────────────────────────────────────
agent_cancel_rate
same_room
country_cancel_rate
lead_time
cancellation_rate
lead_x_no_deposit
adr
total_guests
arrival_date_year
total_nights
stays_in_week_nights
adults
children
stays_in_weekend_nights
is_family
previous_cancellations
adr_per_night
arrival_date_day_of_month
days_in_waiting_list
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "hotel_predictions.db"


# ════════════════════════════════════════════════════════════════
# INIT
# ════════════════════════════════════════════════════════════════

def init_db():

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,

            hotel TEXT,
            market_segment TEXT,
            deposit_type TEXT,
            distribution_channel TEXT,
            customer_type TEXT,

            lead_time INTEGER,
            adr REAL,

            arrival_date_year INTEGER,
            arrival_date_day_of_month INTEGER,

            days_in_waiting_list INTEGER,
            total_of_special_requests INTEGER,
            booking_changes INTEGER,
            is_repeated_guest INTEGER,

            adults INTEGER,
            children INTEGER,
            babies INTEGER,

            stays_in_week_nights INTEGER,
            stays_in_weekend_nights INTEGER,

            previous_cancellations INTEGER,
            previous_bookings_not_canceled INTEGER,

            total_guests REAL,
            is_family INTEGER,

            total_nights INTEGER,
            adr_per_night REAL,

            total_previous_bookings INTEGER,
            cancellation_rate REAL,

            has_booking_changes INTEGER,
            same_room INTEGER,
            request_per_guest REAL,

            lead_x_no_deposit REAL,

            agent_cancel_rate REAL,
            country_cancel_rate REAL,

            features_json TEXT NOT NULL,

            prediction INTEGER NOT NULL,
            cancel_prob REAL NOT NULL
        )
        """)
        conn.commit()

    _migrate_db()


# ════════════════════════════════════════════════════════════════
# MIGRATION
# ════════════════════════════════════════════════════════════════

def _migrate_db():

    COLUMNS = [

        ("market_segment","TEXT","NULL"),
        ("distribution_channel","TEXT","NULL"),
        ("customer_type","TEXT","NULL"),

        ("arrival_date_year","INTEGER","0"),
        ("arrival_date_day_of_month","INTEGER","0"),
        ("days_in_waiting_list","INTEGER","0"),
        ("total_of_special_requests","INTEGER","0"),
        ("booking_changes","INTEGER","0"),
        ("is_repeated_guest","INTEGER","0"),

        ("adults","INTEGER","0"),
        ("children","INTEGER","0"),
        ("babies","INTEGER","0"),
        ("stays_in_week_nights","INTEGER","0"),
        ("stays_in_weekend_nights","INTEGER","0"),

        ("previous_cancellations","INTEGER","0"),
        ("previous_bookings_not_canceled","INTEGER","0"),

        ("total_guests","REAL","0"),
        ("is_family","INTEGER","0"),

        ("total_nights","INTEGER","0"),
        ("adr_per_night","REAL","0"),

        ("total_previous_bookings","INTEGER","0"),
        ("cancellation_rate","REAL","0"),

        ("has_booking_changes","INTEGER","0"),
        ("same_room","INTEGER","0"),
        ("request_per_guest","REAL","0"),

        ("lead_x_no_deposit","REAL","0"),

        ("agent_cancel_rate","REAL","0"),
        ("country_cancel_rate","REAL","0")
    ]

    with sqlite3.connect(DB_PATH) as conn:

        existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}

        for col,typ,default in COLUMNS:

            if col not in existing:

                try:

                    if default=="NULL":
                        conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
                    else:
                        conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ} DEFAULT {default}")

                except sqlite3.OperationalError:
                    pass

        conn.commit()


# ════════════════════════════════════════════════════════════════
# INSERT
# ════════════════════════════════════════════════════════════════

def insert_prediction(

    created_at,

    hotel,
    market_segment,
    deposit_type,
    distribution_channel,
    customer_type,

    lead_time,
    adr,
    arrival_date_year,
    arrival_date_day_of_month,
    days_in_waiting_list,
    total_of_special_requests,
    booking_changes,
    is_repeated_guest,

    adults,
    children,
    babies,

    stays_in_week_nights,
    stays_in_weekend_nights,

    previous_cancellations,
    previous_bookings_not_canceled,

    total_guests,
    is_family,

    total_nights,
    adr_per_night,

    total_previous_bookings,
    cancellation_rate,

    has_booking_changes,
    same_room,
    request_per_guest,

    lead_x_no_deposit,

    agent_cancel_rate,
    country_cancel_rate,

    features_json,

    prediction,
    cancel_prob

):

    with sqlite3.connect(DB_PATH) as conn:

        cur = conn.execute("""

        INSERT INTO predictions (

        created_at,

        hotel,market_segment,deposit_type,
        distribution_channel,customer_type,

        lead_time,adr,

        arrival_date_year,arrival_date_day_of_month,

        days_in_waiting_list,total_of_special_requests,
        booking_changes,is_repeated_guest,

        adults,children,babies,

        stays_in_week_nights,stays_in_weekend_nights,

        previous_cancellations,previous_bookings_not_canceled,

        total_guests,is_family,

        total_nights,adr_per_night,

        total_previous_bookings,cancellation_rate,

        has_booking_changes,same_room,request_per_guest,

        lead_x_no_deposit,

        agent_cancel_rate,country_cancel_rate,

        features_json,

        prediction,cancel_prob

        ) VALUES (

        ?,?,?,?,?,?,
        ?,?,
        ?,?,
        ?,?,
        ?,?,
        ?,?,?,
        ?,?,
        ?,?,
        ?,?,
        ?,?,
        ?,?,
        ?,?,?,
        ?,
        ?,?,
        ?,
        ?,?

        )

        """,

        (

        created_at,

        hotel,market_segment,deposit_type,
        distribution_channel,customer_type,

        int(lead_time),float(adr),

        int(arrival_date_year),int(arrival_date_day_of_month),

        int(days_in_waiting_list),int(total_of_special_requests),
        int(booking_changes),int(is_repeated_guest),

        int(adults),int(children),int(babies),

        int(stays_in_week_nights),int(stays_in_weekend_nights),

        int(previous_cancellations),int(previous_bookings_not_canceled),

        float(total_guests),int(is_family),

        int(total_nights),float(adr_per_night),

        int(total_previous_bookings),float(cancellation_rate),

        int(has_booking_changes),int(same_room),float(request_per_guest),

        float(lead_x_no_deposit),

        float(agent_cancel_rate),float(country_cancel_rate),

        features_json,

        int(prediction),float(cancel_prob)

        )

        )

        conn.commit()

        return cur.lastrowid


# ════════════════════════════════════════════════════════════════
# FETCH LATEST
# ════════════════════════════════════════════════════════════════

def fetch_latest(limit=500):

    with sqlite3.connect(DB_PATH) as conn:

        cur = conn.execute("""

        SELECT

        id,
        created_at,
        hotel,
        lead_time,
        deposit_type,
        market_segment,
        adr,
        total_nights,
        total_guests,
        is_family,
        cancellation_rate,
        same_room,
        request_per_guest,
        prediction,
        cancel_prob

        FROM predictions

        ORDER BY id DESC
        LIMIT ?

        """,(limit,))

        return cur.fetchall()


# ════════════════════════════════════════════════════════════════
# STATS
# ════════════════════════════════════════════════════════════════

def fetch_stats():

    with sqlite3.connect(DB_PATH) as conn:

        cur = conn.execute("""

        SELECT

        COUNT(*),

        SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END),

        ROUND(AVG(cancel_prob)*100,1),

        ROUND(AVG(adr),2),

        ROUND(AVG(lead_time),1),

        ROUND(AVG(total_nights),1),

        ROUND(AVG(cancellation_rate)*100,1),

        SUM(CASE WHEN is_family=1 THEN 1 ELSE 0 END),

        ROUND(

        100.0 * SUM(CASE WHEN prediction=1 THEN 1 ELSE 0 END)
/ NULLIF(COUNT(*),0)

        ,1)

        FROM predictions

        """)

        return cur.fetchone()