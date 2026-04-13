"""
Ride Fare Prediction Model
==========================
Predicts ride fare based on distance, vehicle type, and time of day.

Note on training data
---------------------
The source dataset (Bookings.xlsx) contains 103,024 rides but the fares
in it are uniformly distributed with no correlation to distance or
vehicle type (a 5 km ride costs the same as a 50 km ride). This makes
it impossible to train a useful fare model on that data directly.

To build a meaningful fare prediction model, we generate realistic
training data using typical Indian ride-hailing fare economics
(per-km rates, base fares, surge pricing). The XGBoost model learns
the actual fare structure from this simulation.

The rest of the project (Overview / Analytics tabs, MySQL export)
uses the real Bookings.xlsx dataset.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE = Path(__file__).parent
RNG = np.random.default_rng(42)

# ── Realistic fare structure (Indian ride-hailing market) ─────────────────────
# Each row: (base_fare, rate_per_km, min_fare)
FARE_STRUCTURE = {
    'Bike':        (20, 6,   40),
    'eBike':       (15, 5,   35),
    'Auto':        (30, 12,  60),
    'Mini':        (50, 14,  80),
    'Prime Sedan': (60, 16, 100),
    'Prime Plus':  (70, 18, 120),
    'Prime SUV':   (90, 22, 150),
}
VEHICLE_TYPES = list(FARE_STRUCTURE.keys())


def compute_fare(distance, vehicle, hour, day_of_week):
    """Realistic fare formula with surge, peak hours, and noise."""
    base, rate, min_fare = FARE_STRUCTURE[vehicle]
    fare = base + rate * distance

    # Peak-hour surge (morning/evening rush)
    if hour in (7, 8, 9, 17, 18, 19, 20):
        fare *= 1.25
    # Late-night surge
    if hour >= 22 or hour <= 4:
        fare *= 1.30
    # Weekend surge
    if day_of_week >= 5:
        fare *= 1.10

    # Random noise (traffic, toll, driver variance) ±10%
    fare *= RNG.uniform(0.90, 1.10)
    return max(round(fare, 2), min_fare)


# ── 1. Generate realistic training data ───────────────────────────────────────
print("Generating training data from realistic fare model...")
N = 60_000

sim = pd.DataFrame({
    'Ride_Distance':    RNG.uniform(1, 50, N).round(2),
    'Vehicle_Type':     RNG.choice(VEHICLE_TYPES, N),
    'Hour':             RNG.integers(0, 24, N),
    'DayOfWeek':        RNG.integers(0, 7, N),
})
sim['Booking_Value'] = sim.apply(
    lambda r: compute_fare(r['Ride_Distance'], r['Vehicle_Type'], r['Hour'], r['DayOfWeek']),
    axis=1
)
sim['IsWeekend']  = (sim['DayOfWeek'] >= 5).astype(int)
sim['IsNight']    = ((sim['Hour'] >= 22) | (sim['Hour'] <= 4)).astype(int)
sim['IsPeakHour'] = sim['Hour'].apply(lambda h: 1 if h in (7, 8, 9, 17, 18, 19, 20) else 0)

le_vehicle = LabelEncoder()
le_vehicle.fit(VEHICLE_TYPES)
sim['Vehicle_Type_Enc'] = le_vehicle.transform(sim['Vehicle_Type'])

print(f"  Generated {len(sim):,} synthetic rides")
print(f"  Fare range: Rs.{sim['Booking_Value'].min():.0f} - Rs.{sim['Booking_Value'].max():.0f}")
print(f"  Mean fare:  Rs.{sim['Booking_Value'].mean():.0f}\n")

# ── 2. Define features & target ───────────────────────────────────────────────
FEATURES = [
    'Ride_Distance',
    'Vehicle_Type_Enc',
    'Hour',
    'DayOfWeek',
    'IsWeekend',
    'IsNight',
    'IsPeakHour',
]

X = sim[FEATURES]
y = sim['Booking_Value']

# ── 3. Train / Test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train XGBoost Regressor ────────────────────────────────────────────────
print("Training XGBoost Regressor...")
model = XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
)
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)

# Baseline: always predict the mean fare
baseline_pred = [y_train.mean()] * len(y_test)
baseline_mae  = mean_absolute_error(y_test, baseline_pred)

print("\n" + "=" * 55)
print("        FARE REGRESSION RESULTS")
print("=" * 55)
print(f"  MAE   (Mean Absolute Error)  : Rs. {mae:>7.2f}")
print(f"  RMSE  (Root Mean Sq. Error)  : Rs. {rmse:>7.2f}")
print(f"  R2    (variance explained)   : {r2:>7.4f}  ({r2*100:.1f}%)")
print(f"  Baseline MAE (predict mean)  : Rs. {baseline_mae:>7.2f}")
print(f"  Improvement over baseline    : Rs. {baseline_mae - mae:>7.2f}")
print("=" * 55)

if r2 >= 0.85:
    print("  [EXCELLENT] Model explains >85% of fare variance")
elif r2 >= 0.60:
    print("  [GOOD] Model explains 60-85% of fare variance")
elif r2 >= 0.30:
    print("  [FAIR] Model explains 30-60% of fare variance")
else:
    print("  [POOR] Model cannot predict fare well")
print("=" * 55)

print("\nFeature Importances:")
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, score in fi.items():
    bar = "#" * int(score * 60)
    print(f"  {feat:<20} {bar}  {score:.4f}")

# ── 6. Sanity check — show a few real predictions ─────────────────────────────
print("\nSample predictions:")
samples = [
    (5,  'Bike',        9,  1),
    (20, 'Auto',        14, 2),
    (35, 'Prime Sedan', 18, 4),
    (50, 'Prime SUV',   23, 5),
]
for dist, veh, hr, dow in samples:
    vt_enc = int(le_vehicle.transform([veh])[0])
    is_w = 1 if dow >= 5 else 0
    is_n = 1 if (hr >= 22 or hr <= 4) else 0
    is_p = 1 if hr in (7, 8, 9, 17, 18, 19, 20) else 0
    row = pd.DataFrame([{
        'Ride_Distance': dist, 'Vehicle_Type_Enc': vt_enc, 'Hour': hr,
        'DayOfWeek': dow, 'IsWeekend': is_w, 'IsNight': is_n, 'IsPeakHour': is_p,
    }])
    pred = model.predict(row)[0]
    print(f"  {dist:>2}km  {veh:<12} {hr:02d}:00 -> Rs. {pred:>7.2f}")

# ── 7. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(model,       BASE / "xgb_model.pkl")
joblib.dump(le_vehicle,  BASE / "le_vehicle.pkl")
joblib.dump(FEATURES,    BASE / "features.pkl")
joblib.dump(VEHICLE_TYPES, BASE / "vehicle_types.pkl")
print("\nModel artifacts saved.")

# ── 8. Save predictions CSV (test set) ────────────────────────────────────────
results = X_test.copy()
results['Vehicle_Type']   = le_vehicle.inverse_transform(results['Vehicle_Type_Enc'].astype(int))
results['Actual_Fare']    = y_test.values.round(2)
results['Predicted_Fare'] = y_pred.round(2)
results['Error']          = (results['Actual_Fare'] - results['Predicted_Fare']).round(2)
results['Abs_Error']      = results['Error'].abs()
results.to_csv(BASE / "predictions.csv", index=False)
print("Test predictions saved -> predictions.csv")

# ── 9. Save cleaned data from REAL dataset for dashboard analytics ────────────
print("\nLoading real dataset for dashboard analytics...")
real = pd.read_excel(BASE / "Bookings.xlsx")
real['Date'] = pd.to_datetime(real['Date'], errors='coerce')
real['Hour'] = pd.to_datetime(real['Time'], format='%H:%M:%S', errors='coerce').dt.hour
real['DayOfWeek']  = real['Date'].dt.dayofweek
real['IsWeekend']  = (real['DayOfWeek'] >= 5).astype(int)
real['IsNight']    = ((real['Hour'] >= 22) | (real['Hour'] <= 5)).astype(int)
real['IsPeakHour'] = real['Hour'].apply(lambda h: 1 if h in (7, 8, 9, 17, 18, 19, 20) else 0)
real['Driver_Ratings']  = real['Driver_Ratings'].fillna(real['Driver_Ratings'].median())
real['Customer_Rating'] = real['Customer_Rating'].fillna(real['Customer_Rating'].median())
real['Ride_Distance']   = real['Ride_Distance'].fillna(0)

df_save = real[['Date', 'Hour', 'DayOfWeek', 'IsWeekend', 'IsNight', 'IsPeakHour',
                'Vehicle_Type', 'Booking_Value', 'Ride_Distance',
                'Driver_Ratings', 'Customer_Rating', 'Booking_Status']].copy()
df_save.to_csv(BASE / "cleaned_data.csv", index=False)
print("Cleaned data saved -> cleaned_data.csv")
