# Ride Fare Prediction

Machine learning project that predicts ride fare based on distance, vehicle type, and time of day.

Built with XGBoost, Streamlit, and MySQL.

## What it does

Given a ride's distance, vehicle type, hour, and day of week, the model predicts the fare in Rs. The dashboard also shows analytics on 103,024 real ride bookings from July 2024.

## Tech stack

- **Model:** XGBoost Regressor
- **Dashboard:** Streamlit + Plotly
- **Database:** MySQL (via MySQL Workbench)

## Data note

The real dataset (`Bookings.xlsx`) contains 103,024 rides but the fares in it are uniformly distributed — a 5 km ride costs the same as a 50 km ride. Because of that, no model can learn a useful fare signal from the raw data.

To build a **meaningful** fare prediction model, the training data is simulated from a realistic Indian ride-hailing fare formula (base fare + per-km rate + surge pricing for peak hours, late night, and weekends). The XGBoost model learns the actual fare structure from this simulation.

The real dataset is still used throughout the **Overview** and **Analytics** tabs to show actual booking patterns, vehicle mix, cancellation rates, and so on.

## Model performance

- **R² = 0.989** (98.9% variance explained)
- **MAE = Rs. 24** (average absolute error)
- **RMSE = Rs. 34**

Top features by importance:
1. Vehicle type (52.6%)
2. Distance (30.9%)
3. Late-night surge (7.9%)
4. Peak-hour surge (6.8%)

## Project structure

```
.
├── Bookings.xlsx       # Raw dataset (103,024 rides)
├── model.py            # Generate training data + train XGBoost regressor
├── app.py              # Streamlit dashboard (Overview, Analytics, Model, Predict)
├── mysql_push.py       # Push cleaned data to MySQL Workbench
├── requirements.txt
├── .env.example        # Template for MySQL credentials
└── .gitignore
```

## Setup

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

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```
   Opens at http://localhost:8501

4. **(Optional) Push to MySQL**
   - Copy `.env.example` to `.env` and fill in your MySQL credentials
   - Make sure MySQL Workbench is running
   ```bash
   python mysql_push.py
   ```

## Dashboard pages

- **Overview** — ride totals, status distribution, vehicle breakdown, success rate, hourly/daily patterns (from real data)
- **Analytics** — fare and distance distributions, vehicle comparison radar, scatter plots (from real data)
- **Model** — regression metrics (R², MAE, RMSE), actual vs predicted scatter, residual distribution, error by vehicle type
- **Predict** — live fare prediction: pick vehicle, distance, hour, and day → see predicted fare, comparison across all vehicle types, fare-vs-distance curve, and fare-by-hour chart

## Security

`.env` (MySQL password) is listed in `.gitignore` and never committed. Use `.env.example` as a template.
