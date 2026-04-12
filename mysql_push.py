"""
MySQL Workbench Integration
============================
Creates the `ride_bookings` database and pushes:
  - bookings        : cleaned full dataset
  - predictions     : model test predictions
  - model_metrics   : accuracy, F1 scores

Usage:
  1. Open MySQL Workbench and ensure your server is running.
  2. Add your credentials to .env (already set up — never commit that file).
  3. Run:  python mysql_push.py
"""

import os
import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG — loaded from .env (never hardcoded) ───────────────────────────────
HOST     = os.getenv("MYSQL_HOST",     "localhost")
PORT     = int(os.getenv("MYSQL_PORT", "3306"))
USER     = os.getenv("MYSQL_USER",     "root")
PASSWORD = os.getenv("MYSQL_PASSWORD", "")
DATABASE = os.getenv("MYSQL_DATABASE", "ride_bookings")

BASE = "c:/Users/Utkarsha/Downloads/Rides"
# ─────────────────────────────────────────────────────────────────────────────


def get_connection(database=None):
    return mysql.connector.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=database,
    )


def create_database(cursor):
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DATABASE}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    print(f"  Database `{DATABASE}` ready.")


def create_tables(cursor):
    cursor.execute(f"USE `{DATABASE}`;")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            booking_date    DATE,
            hour            INT,
            day_of_week     INT,
            is_weekend      TINYINT(1),
            is_night        TINYINT(1),
            is_peak_hour    TINYINT(1),
            vehicle_type    VARCHAR(50),
            booking_value   DECIMAL(10,2),
            ride_distance   DECIMAL(10,2),
            driver_rating   DECIMAL(3,1),
            customer_rating DECIMAL(3,1),
            booking_status  VARCHAR(50)
        ) ENGINE=InnoDB;
    """)
    print("  Table `bookings` ready.")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            vehicle_type_enc INT,
            pickup_enc       INT,
            drop_enc         INT,
            booking_value    DECIMAL(10,2),
            hour             INT,
            day_of_week      INT,
            is_weekend       TINYINT(1),
            is_night         TINYINT(1),
            is_peak_hour     TINYINT(1),
            actual_status    INT,
            predicted_status INT,
            actual_label     VARCHAR(50),
            predicted_label  VARCHAR(50),
            correct          TINYINT(1)
        ) ENGINE=InnoDB;
    """)
    print("  Table `predictions` ready.")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id           INT AUTO_INCREMENT PRIMARY KEY,
            run_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_name   VARCHAR(100),
            accuracy     DECIMAL(6,4),
            f1_macro     DECIMAL(6,4),
            f1_weighted  DECIMAL(6,4),
            notes        TEXT
        ) ENGINE=InnoDB;
    """)
    print("  Table `model_metrics` ready.")


def push_bookings(cursor, conn):
    df = pd.read_csv(f"{BASE}/cleaned_data.csv", parse_dates=['Date'])

    # Fill missing ratings with median of successful rides
    median_driver   = df.loc[df['Booking_Status'] == 'Success', 'Driver_Ratings'].median()
    median_customer = df.loc[df['Booking_Status'] == 'Success', 'Customer_Rating'].median()
    df['Driver_Ratings']  = df['Driver_Ratings'].fillna(median_driver)
    df['Customer_Rating'] = df['Customer_Rating'].fillna(median_customer)

    # Fill missing ride distance with 0 for canceled rides
    df['Ride_Distance'] = df['Ride_Distance'].fillna(0)

    df = df.where(pd.notnull(df), None)

    # Clear existing data
    cursor.execute("TRUNCATE TABLE bookings;")

    insert_sql = """
        INSERT INTO bookings
          (booking_date, hour, day_of_week, is_weekend, is_night, is_peak_hour,
           vehicle_type, booking_value, ride_distance, driver_rating, customer_rating, booking_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    rows = []
    for _, row in df.iterrows():
        rows.append((
            row['Date'].date() if pd.notnull(row['Date']) else None,
            int(row['Hour'])       if pd.notnull(row['Hour'])       else None,
            int(row['DayOfWeek'])  if pd.notnull(row['DayOfWeek'])  else None,
            int(row['IsWeekend'])  if pd.notnull(row['IsWeekend'])  else None,
            int(row['IsNight'])    if pd.notnull(row['IsNight'])     else None,
            int(row['IsPeakHour']) if pd.notnull(row['IsPeakHour']) else None,
            row['Vehicle_Type'],
            float(row['Booking_Value'])   if pd.notnull(row['Booking_Value'])   else None,
            float(row['Ride_Distance'])   if pd.notnull(row['Ride_Distance'])   else None,
            float(row['Driver_Ratings'])  if pd.notnull(row['Driver_Ratings'])  else None,
            float(row['Customer_Rating']) if pd.notnull(row['Customer_Rating']) else None,
            row['Booking_Status'],
        ))

    # Batch insert in chunks of 5000
    chunk = 5000
    for i in range(0, len(rows), chunk):
        cursor.executemany(insert_sql, rows[i:i + chunk])
        conn.commit()

    print(f"  Inserted {len(rows):,} rows into `bookings`.")


def push_predictions(cursor, conn):
    pred = pd.read_csv(f"{BASE}/predictions.csv")
    pred = pred.where(pd.notnull(pred), None)

    cursor.execute("TRUNCATE TABLE predictions;")

    insert_sql = """
        INSERT INTO predictions
          (vehicle_type_enc, pickup_enc, drop_enc, booking_value, hour,
           day_of_week, is_weekend, is_night, is_peak_hour,
           actual_status, predicted_status, actual_label, predicted_label, correct)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    rows = [
        (
            int(r['Vehicle_Type_Enc']), int(r['Pickup_Enc']), int(r['Drop_Enc']),
            float(r['Booking_Value']), int(r['Hour']), int(r['DayOfWeek']),
            int(r['IsWeekend']), int(r['IsNight']), int(r['IsPeakHour']),
            int(r['Actual_Status']), int(r['Predicted_Status']),
            r['Actual_Label'], r['Predicted_Label'], int(r['Correct']),
        )
        for _, r in pred.iterrows()
    ]

    chunk = 5000
    for i in range(0, len(rows), chunk):
        cursor.executemany(insert_sql, rows[i:i + chunk])
        conn.commit()

    print(f"  Inserted {len(rows):,} rows into `predictions`.")


def push_metrics(cursor, conn):
    from sklearn.metrics import accuracy_score, f1_score
    pred = pd.read_csv(f"{BASE}/predictions.csv")
    acc    = accuracy_score(pred['Actual_Status'], pred['Predicted_Status'])
    f1_mac = f1_score(pred['Actual_Status'], pred['Predicted_Status'], average='macro')
    f1_wt  = f1_score(pred['Actual_Status'], pred['Predicted_Status'], average='weighted')

    cursor.execute("""
        INSERT INTO model_metrics (model_name, accuracy, f1_macro, f1_weighted, notes)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        'XGBoost Classifier',
        round(acc, 4),
        round(f1_mac, 4),
        round(f1_wt, 4),
        'Synthetic dataset — uniform class distribution across features. '
        'Real-world data would yield higher accuracy.',
    ))
    conn.commit()
    print(f"  Model metrics saved (Accuracy={acc:.4f}, F1_macro={f1_mac:.4f}).")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Connecting to MySQL...")
    try:
        conn = get_connection()
        cursor = conn.cursor()

        print("Creating database...")
        create_database(cursor)
        conn.database = DATABASE

        print("Creating tables...")
        create_tables(cursor)

        print("Pushing bookings data...")
        push_bookings(cursor, conn)

        print("Pushing predictions...")
        push_predictions(cursor, conn)

        print("Saving model metrics...")
        push_metrics(cursor, conn)

        cursor.close()
        conn.close()
        print("\nDone! All data is now in MySQL Workbench.")
        print(f"  Host: {HOST}:{PORT}  Database: {DATABASE}")

    except Error as e:
        print(f"\nMySQL Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure MySQL Server is running in MySQL Workbench")
        print("  2. Update HOST / USER / PASSWORD at the top of this file")
        print("  3. Run again: python mysql_push.py")
