# FareCast

A machine learning app that predicts ride fare based on distance, vehicle type, and time of day.

**Live demo:** https://farecast-ride.streamlit.app/

Built with XGBoost, Streamlit, Plotly, and MySQL (Railway).

## What it does

Given a ride's distance, vehicle type, hour, and day of week, the model predicts the fare in Rs. Every prediction can be saved to a cloud MySQL database. The dashboard also shows analytics on 103,024 real ride bookings from July 2024.

## Tech stack

- **Model:** XGBoost Regressor
- **Dashboard:** Streamlit + Plotly
- **Database:** MySQL hosted on Railway
- **Deployment:** Streamlit Cloud (auto-deploys from GitHub)

## Model performance

- **R² = 0.989** (98.9% variance explained)
- **MAE = Rs. 24** (average absolute error)
- **RMSE = Rs. 34**

Top features by importance:

1. Vehicle type (52.6%)
2. Distance (30.9%)
3. Late-night surge (7.9%)
4. Peak-hour surge (6.8%)

## Data note

The real dataset (`Bookings.xlsx`) contains 103,024 rides but the fares in it are uniformly distributed — a 5 km ride costs the same as a 50 km ride. Because of that, no model can learn a useful fare signal from the raw data.

To build a **meaningful** fare prediction model, the training data is simulated from a realistic Indian ride-hailing fare formula (base fare + per-km rate + surge pricing for peak hours, late night, and weekends). The XGBoost model learns the actual fare structure from this simulation.

The real dataset is still used throughout the **Insights** tab to show actual booking patterns, vehicle mix, distance distribution, and cancellation reasons.

## Dashboard pages

- **Predict** — pick vehicle, distance, hour, and day → see predicted fare, fare curve across distances, and a comparison across all vehicle types. Save any prediction to MySQL with one click.
- **Insights** — KPIs + 6 charts: booking status, ride distance distribution, cancellation reasons, actual vs predicted scatter, feature importances, and residual distribution.

## Project structure

```
.
├── Bookings.xlsx       # Raw dataset (103,024 rides)
├── model.py            # Generate training data + train XGBoost regressor
├── app.py              # Streamlit dashboard (Predict + Insights tabs)
├── db.py               # MySQL connection + save_prediction helper
├── requirements.txt
├── .env.example        # Template for MySQL credentials
└── .gitignore
```

## Setup (local)

1. **Clone and install**
   ```bash
   git clone https://github.com/utkarsha-20/ride-booking-ml.git
   cd ride-booking-ml
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python model.py
   ```
   Generates the training data, trains XGBoost, and saves `xgb_model.pkl`, `predictions.csv`, and `cleaned_data.csv`.

3. **Set up MySQL credentials** (optional — for saving predictions)

   Copy `.env.example` to `.env` and fill in your MySQL connection details. Any MySQL host works (Railway, Filess.io, local MySQL Workbench, etc.).
   ```
   MYSQL_HOST=your-host
   MYSQL_PORT=3306
   MYSQL_USER=root
   MYSQL_PASSWORD=your-password
   MYSQL_DATABASE=railway
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

## Security

- `.env` is in `.gitignore` and never committed
- On Streamlit Cloud, the same credentials live in the app's **Secrets** panel (not in git)
- `db.py` reads from Streamlit secrets first, then falls back to `.env`

## Screenshots

Live app: https://farecast-ride.streamlit.app/
