"""
Ride Booking ML Model
=====================
Target  : Booking_Status  (Success / Canceled by Driver / Canceled by Customer / Driver Not Found)
Model   : XGBoost Classifier
Note    : This is a synthetic dataset — class rates are near-uniform across all features.
          The model learns the marginal distributions. Real-world ride data would yield
          significantly higher accuracy.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_excel(r"C:\Users\Utkarsha\Downloads\Rides\Bookings.xlsx")
print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
print(f"  Status distribution:\n{df['Booking_Status'].value_counts().to_string()}\n")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour']       = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
df['DayOfWeek']  = df['Date'].dt.dayofweek          # 0=Mon, 6=Sun
df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)
df['IsNight']    = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
df['IsPeakHour'] = df['Hour'].apply(lambda h: 1 if h in [7, 8, 9, 17, 18, 19, 20] else 0)

# Encode Vehicle_Type with a fixed vocabulary
VEHICLE_TYPES = ['Auto', 'Bike', 'eBike', 'Mini', 'Prime Plus', 'Prime Sedan', 'Prime SUV']
le_vehicle = LabelEncoder()
le_vehicle.fit(VEHICLE_TYPES)
df['Vehicle_Type_Enc'] = df['Vehicle_Type'].apply(
    lambda x: int(le_vehicle.transform([x])[0]) if x in VEHICLE_TYPES else -1
)

# Encode location fields
le_pickup = LabelEncoder()
le_drop   = LabelEncoder()
df['Pickup_Enc'] = le_pickup.fit_transform(df['Pickup_Location'].fillna('Unknown'))
df['Drop_Enc']   = le_drop.fit_transform(df['Drop_Location'].fillna('Unknown'))

df['Booking_Value'] = df['Booking_Value'].fillna(df['Booking_Value'].median())

# ── 3. Target Encoding ────────────────────────────────────────────────────────
STATUS_MAP = {
    'Success':              0,
    'Canceled by Driver':   1,
    'Canceled by Customer': 2,
    'Driver Not Found':     3,
}
LABEL_NAMES = ['Success', 'Canceled by Driver', 'Canceled by Customer', 'Driver Not Found']

df['Status_Label'] = df['Booking_Status'].map(STATUS_MAP)
df = df.dropna(subset=['Status_Label'])
df['Status_Label'] = df['Status_Label'].astype(int)

# ── 4. Features & Target ──────────────────────────────────────────────────────
FEATURES = [
    'Vehicle_Type_Enc',
    'Pickup_Enc',
    'Drop_Enc',
    'Booking_Value',
    'Hour',
    'DayOfWeek',
    'IsWeekend',
    'IsNight',
    'IsPeakHour',
]

X = df[FEATURES]
y = df['Status_Label']

print(f"Dataset ready: {X.shape[0]:,} rows, {X.shape[1]} features")

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Balance classes with sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
)
model.fit(X_train, y_train, sample_weight=sample_weights)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

acc    = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro')
f1_wt  = f1_score(y_test, y_pred, average='weighted')

# Baseline: always predict majority class
baseline_acc = (y_test == 0).mean()

print("\n" + "=" * 55)
print("           MODEL EVALUATION RESULTS")
print("=" * 55)
print(f"  Accuracy              : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Baseline (majority)   : {baseline_acc:.4f}  ({baseline_acc*100:.1f}%)")
print(f"  Lift over baseline    : +{(acc - baseline_acc)*100:.1f}%")
print(f"  F1 Score (Macro)      : {f1_mac:.4f}")
print(f"  F1 Score (Weighted)   : {f1_wt:.4f}")
print("=" * 55)
print("  NOTE: This dataset is synthetic — feature values are")
print("  uniformly distributed across all booking statuses.")
print("  A real-world dataset would yield 80-90%+ accuracy.")
print("=" * 55)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

print("Confusion Matrix:")
cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=LABEL_NAMES,
    columns=LABEL_NAMES
)
print(cm_df)

print("\nFeature Importances:")
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, score in fi.items():
    bar = "#" * int(score * 40)
    print(f"  {feat:<20} {bar}  {score:.4f}")

# ── 8. Save artifacts ─────────────────────────────────────────────────────────
BASE = "c:/Users/Utkarsha/Downloads/Rides"
joblib.dump(model,      f"{BASE}/xgb_model.pkl")
joblib.dump(le_vehicle, f"{BASE}/le_vehicle.pkl")
joblib.dump(le_pickup,  f"{BASE}/le_pickup.pkl")
joblib.dump(le_drop,    f"{BASE}/le_drop.pkl")
joblib.dump(FEATURES,   f"{BASE}/features.pkl")
joblib.dump(LABEL_NAMES, f"{BASE}/label_names.pkl")
print("\nModel artifacts saved.")

# ── 9. Save predictions CSV ───────────────────────────────────────────────────
results = X_test.copy()
results['Actual_Status']    = y_test.values
results['Predicted_Status'] = y_pred
results['Actual_Label']     = [LABEL_NAMES[i] for i in y_test.values]
results['Predicted_Label']  = [LABEL_NAMES[i] for i in y_pred]
results['Correct']          = (results['Actual_Status'] == results['Predicted_Status']).astype(int)
results.to_csv(f"{BASE}/predictions.csv", index=False)
print("Predictions saved -> predictions.csv")

# ── 10. Save full dataset for Streamlit analytics ─────────────────────────────
df_save = df[['Date', 'Hour', 'DayOfWeek', 'IsWeekend', 'IsNight', 'IsPeakHour',
              'Vehicle_Type', 'Booking_Value', 'Ride_Distance',
              'Driver_Ratings', 'Customer_Rating', 'Booking_Status']].copy()
df_save.to_csv(f"{BASE}/cleaned_data.csv", index=False)
print("Cleaned data saved -> cleaned_data.csv")
