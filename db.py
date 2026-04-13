"""
MySQL connection helper.

Reads credentials from Streamlit secrets (when deployed) or .env (local).
Exposes a connection factory and a function to create the
fare_predictions table.
"""
import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")


def _cfg(key, default=None):
    """Get config from Streamlit secrets first, then env, then default."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


def get_connection():
    return mysql.connector.connect(
        host=_cfg("MYSQL_HOST"),
        port=int(_cfg("MYSQL_PORT", "3306")),
        user=_cfg("MYSQL_USER"),
        password=_cfg("MYSQL_PASSWORD"),
        database=_cfg("MYSQL_DATABASE"),
        connection_timeout=10,
    )


def ensure_schema():
    """Create the fare_predictions table if it does not exist."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fare_predictions (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                vehicle_type    VARCHAR(50)  NOT NULL,
                ride_distance   DECIMAL(5,1) NOT NULL,
                hour            INT          NOT NULL,
                day_of_week     VARCHAR(10)  NOT NULL,
                is_weekend      TINYINT(1)   NOT NULL,
                is_night        TINYINT(1)   NOT NULL,
                is_peak_hour    TINYINT(1)   NOT NULL,
                predicted_fare  DECIMAL(8,2) NOT NULL
            ) ENGINE=InnoDB;
        """)
        conn.commit()
    finally:
        conn.close()


def save_prediction(
    vehicle_type: str,
    ride_distance: float,
    hour: int,
    day_of_week: str,
    is_weekend: int,
    is_night: int,
    is_peak_hour: int,
    predicted_fare: float,
) -> int:
    """Insert a fare prediction row and return its new id."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fare_predictions
                (vehicle_type, ride_distance, hour, day_of_week,
                 is_weekend, is_night, is_peak_hour, predicted_fare)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                vehicle_type,
                float(ride_distance),
                int(hour),
                day_of_week,
                int(is_weekend),
                int(is_night),
                int(is_peak_hour),
                float(predicted_fare),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def fetch_recent(limit: int = 10):
    """Return the most recent prediction rows as a list of dicts."""
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT id, created_at, vehicle_type, ride_distance,
                   hour, day_of_week, predicted_fare
            FROM fare_predictions
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()
    finally:
        conn.close()
