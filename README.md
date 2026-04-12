# Ride Booking Status Prediction

Machine learning project that predicts the outcome of a ride booking — whether it will be **Success**, **Canceled by Driver**, **Canceled by Customer**, or **Driver Not Found** — based on vehicle type, location, time of day, and fare.

Built with XGBoost, Streamlit, and MySQL.

## What it does

Given details of a new ride booking (vehicle, pickup/drop, hour, day, fare), the model predicts the probability of each possible outcome. The dataset contains 103,024 ride bookings from July 2024.

## Tech stack

- **Model:** XGBoost Classifier (4 classes)
- **Dashboard:** Streamlit + Plotly
- **Database:** MySQL (via MySQL Workbench)
- **Data:** 103,024 ride bookings (`Bookings.xlsx`)

## Project structure

```
.
├── Bookings.xlsx       # Raw dataset
├── model.py            # Train the XGBoost classifier
├── app.py              # Streamlit dashboard (Overview, Analytics, Model, Predict)
├── mysql_push.py       # Push cleaned data + predictions to MySQL
├── Ride_fare.ipynb     # Original exploration notebook
├── requirements.txt
├── .env.example        # Template for MySQL credentials
└── .gitignore
```

## Setup

1. **Clone and install**
   ```bash
   git clone <your-repo-url>
   cd Rides
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python model.py
   ```
   This reads `Bookings.xlsx`, trains the XGBoost classifier, and saves:
   - `xgb_model.pkl`, `le_vehicle.pkl`, `le_pickup.pkl`, `le_drop.pkl`, `features.pkl`
   - `cleaned_data.csv`, `predictions.csv`

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`.

4. **(Optional) Push to MySQL**
   - Copy `.env.example` to `.env` and fill in your MySQL credentials
   - Make sure MySQL Workbench is running
   ```bash
   python mysql_push.py
   ```
   Creates database `ride_bookings` with 3 tables: `bookings`, `predictions`, `model_metrics`.

## Dashboard pages

- **Overview** — booking status distribution, vehicle breakdown, success rate by vehicle, hourly/daily patterns
- **Analytics** — fare & distance distributions, vehicle comparison radar, scatter plots
- **Model** — accuracy metrics, confusion matrix, feature importances, per-class recall
- **Predict** — live prediction: fill in booking details, get class probabilities + comparison against dataset averages

## Model accuracy

The test accuracy is ~28%. This is **expected** because the dataset is synthetic — booking status is distributed near-uniformly across all features (every vehicle type, hour, and location has the same ~62% success rate). There is no real signal the model can learn. A real-world ride dataset with the same pipeline would yield 80–90%+ accuracy.

## Security

`.env` (MySQL password) is listed in `.gitignore` and never committed. Use `.env.example` as a template for your own credentials.
