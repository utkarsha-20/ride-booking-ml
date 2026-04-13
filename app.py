"""
Ride Booking Analytics & ML Dashboard
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Ride Bookings",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Void Space palette ────────────────────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b22"
PRIMARY  = "#58a6ff"
TEXT     = "#c9d1d9"
MUTED    = "#8b949e"
BORDER   = "#30363d"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"

STATUS_COLORS = {
    'Success':              GREEN,
    'Canceled by Driver':   RED,
    'Canceled by Customer': ORANGE,
    'Driver Not Found':     PURPLE,
}

LABEL_NAMES = ['Success', 'Canceled by Driver', 'Canceled by Customer', 'Driver Not Found']
CH = 240  # chart height

st.markdown(f"""
<style>
    /* Hide sidebar toggle */
    [data-testid="collapsedControl"] {{ display: none; }}

    .stApp {{
        background: {BG};
    }}

    .block-container {{
        padding: 0.6rem 1.2rem 0.4rem 1.2rem !important;
        max-width: 100% !important;
    }}

    /* Top bar */
    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid {BORDER};
        padding: 0 0 8px 0;
        margin-bottom: 8px;
    }}
    .topbar-title {{
        font-size: 14px;
        font-weight: 600;
        color: {TEXT};
        letter-spacing: -0.3px;
    }}
    .topbar-meta {{
        font-size: 11px;
        color: {MUTED};
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 6px;
        padding: 8px 12px !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.68rem !important;
        color: {MUTED} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.1rem !important;
        color: {TEXT} !important;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0px;
        border-bottom: 1px solid {BORDER};
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 36px;
        padding: 0 16px;
        font-size: 13px;
        color: {MUTED};
        border-radius: 0;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT} !important;
        border-bottom: 2px solid {PRIMARY};
        background: transparent !important;
    }}

    /* Chart containers */
    .stPlotlyChart {{
        margin-bottom: -6px !important;
    }}

    /* Info box */
    .stAlert {{
        padding: 6px 10px !important;
        font-size: 0.75rem !important;
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
    }}

    hr {{ margin: 4px 0 !important; border-color: {BORDER} !important; }}

    /* Expander */
    .streamlit-expanderHeader {{
        font-size: 0.8rem !important;
        color: {MUTED} !important;
    }}

    /* Remove extra gaps */
    [data-testid="stVerticalBlock"] > div {{
        gap: 0.25rem !important;
    }}

    /* Neutral button — override primary red/orange */
    .stButton > button {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        padding: 6px 14px !important;
        box-shadow: none !important;
    }}
    .stButton > button:hover {{
        background: #222831 !important;
        border-color: #484f58 !important;
        color: {TEXT} !important;
    }}
    .stButton > button:focus {{
        box-shadow: none !important;
        outline: none !important;
    }}

    /* Form inputs — consistent neutral styling */
    [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <span class="topbar-title">Ride Fare Prediction</span>
    <span class="topbar-meta">103,024 records &middot; July 2024 &middot; XGBoost Regressor</span>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
import runpy
from pathlib import Path

BASE = Path(__file__).parent

# If model artifacts don't exist, train the model first (runs once on deploy)
if not (BASE / "xgb_model.pkl").exists():
    with st.spinner("First-time setup: training model (takes ~30 seconds)..."):
        try:
            runpy.run_path(str(BASE / "model.py"), run_name="__main__")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

def _mtime(path):
    return path.stat().st_mtime if path.exists() else 0

@st.cache_data
def load_data(_mtime_key):
    return pd.read_csv(BASE / "cleaned_data.csv", parse_dates=['Date'])

@st.cache_data
def load_predictions(_mtime_key):
    return pd.read_csv(BASE / "predictions.csv")

@st.cache_resource
def load_model(_mtime_key):
    model    = joblib.load(BASE / "xgb_model.pkl")
    features = joblib.load(BASE / "features.pkl")
    le_v     = joblib.load(BASE / "le_vehicle.pkl")
    return model, features, le_v

df   = load_data(_mtime(BASE / "cleaned_data.csv"))
pred = load_predictions(_mtime(BASE / "predictions.csv"))
model, features, le_vehicle = load_model(_mtime(BASE / "xgb_model.pkl"))

VEHICLE_TYPES = ['Bike', 'eBike', 'Auto', 'Mini', 'Prime Sedan', 'Prime Plus', 'Prime SUV']

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=TEXT, size=10),
    margin=dict(t=28, b=20, l=40, r=15),
    height=CH,
)

def apply_layout(fig, title="", **overrides):
    layout = {**CHART_LAYOUT, **overrides}
    if title:
        layout['title'] = dict(text=title, font=dict(size=11, color=TEXT), x=0, xanchor='left')
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=9))
    return fig


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_insights = st.tabs(["Predict", "Insights"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT — Fare calculator
# ═════════════════════════════════════════════════════════════════════════════
with tab_predict:
    HOUR_OPTIONS = [f"{h:02d}:00" for h in range(24)]
    DAY_OPTIONS  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday', 'Sunday']
    DOW_MAP = {d: i for i, d in enumerate(DAY_OPTIONS)}

    # ── Labels row ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:grid;
                grid-template-columns: 1.4fr 1fr 1fr 1.3fr 0.9fr;
                gap:12px; padding:4px 0 4px 0;">
        <span style="font-size:11px; color:{MUTED};">Vehicle type</span>
        <span style="font-size:11px; color:{MUTED};">Distance (km)</span>
        <span style="font-size:11px; color:{MUTED};">Hour</span>
        <span style="font-size:11px; color:{MUTED};">Day of week</span>
        <span></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Inputs row ────────────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([1.4, 1, 1, 1.3, 0.9])
    with f1:
        vehicle_type = st.selectbox(
            "Vehicle", VEHICLE_TYPES, index=4,
            label_visibility="collapsed",
        )
    with f2:
        distance = st.number_input(
            "Distance", min_value=1, max_value=50, value=15, step=1,
            label_visibility="collapsed",
        )
    with f3:
        hour_label = st.selectbox(
            "Hour", HOUR_OPTIONS, index=9,
            label_visibility="collapsed",
        )
        hour = int(hour_label.split(":")[0])
    with f4:
        day_of_week = st.selectbox(
            "Day", DAY_OPTIONS, index=1,
            label_visibility="collapsed",
        )
        dow = DOW_MAP[day_of_week]
    with f5:
        predict_btn = st.button("Calculate fare", use_container_width=True)

    # Thin separator
    st.markdown(f"<hr style='border:none; border-top:1px solid {BORDER}; margin:12px 0 14px 0;'/>",
                unsafe_allow_html=True)

    if predict_btn:
        is_weekend = 1 if dow >= 5 else 0
        is_night   = 1 if (hour >= 22 or hour <= 4) else 0
        is_peak    = 1 if hour in (7, 8, 9, 17, 18, 19, 20) else 0

        vt_enc = int(le_vehicle.transform([vehicle_type])[0])

        input_df = pd.DataFrame([{
            'Ride_Distance':    distance,
            'Vehicle_Type_Enc': vt_enc,
            'Hour':             hour,
            'DayOfWeek':        dow,
            'IsWeekend':        is_weekend,
            'IsNight':          is_night,
            'IsPeakHour':       is_peak,
        }])

        predicted_fare = float(model.predict(input_df)[0])

        # Left side: definition-list style result.
        # Right side: fare curve chart.
        left, right = st.columns([1, 1.6])

        with left:
            surge_parts = []
            if is_peak:    surge_parts.append("peak hour")
            if is_night:   surge_parts.append("late night")
            if is_weekend: surge_parts.append("weekend")
            surge_text = ", ".join(surge_parts) if surge_parts else "none"

            st.markdown(f"""
            <div style="display:flex; align-items:baseline; gap:8px;
                        padding-bottom:12px; border-bottom:1px solid {BORDER};
                        margin-bottom:12px;">
                <span style="font-size:13px; color:{MUTED};">Fare</span>
                <span style="font-size:26px; font-weight:600; color:{TEXT}; line-height:1;">
                    Rs. {predicted_fare:,.0f}
                </span>
            </div>

            <table style="width:100%; border-collapse:collapse; font-size:13px; color:{TEXT};">
                <tr>
                    <td style="padding:6px 0; color:{MUTED}; width:42%;">Vehicle</td>
                    <td style="padding:6px 0;">{vehicle_type}</td>
                </tr>
                <tr>
                    <td style="padding:6px 0; color:{MUTED};
                               border-top:1px solid {BORDER};">Distance</td>
                    <td style="padding:6px 0; border-top:1px solid {BORDER};">{distance} km</td>
                </tr>
                <tr>
                    <td style="padding:6px 0; color:{MUTED};
                               border-top:1px solid {BORDER};">Time</td>
                    <td style="padding:6px 0; border-top:1px solid {BORDER};">
                        {day_of_week}, {hour_label}
                    </td>
                </tr>
                <tr>
                    <td style="padding:6px 0; color:{MUTED};
                               border-top:1px solid {BORDER};">Surge</td>
                    <td style="padding:6px 0; border-top:1px solid {BORDER};">{surge_text}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)

        with right:
            # Fare curve across all distances for selected vehicle + time
            curve_rows = []
            for d in range(1, 51):
                row = pd.DataFrame([{
                    'Ride_Distance':    d,
                    'Vehicle_Type_Enc': vt_enc,
                    'Hour':             hour,
                    'DayOfWeek':        dow,
                    'IsWeekend':        is_weekend,
                    'IsNight':          is_night,
                    'IsPeakHour':       is_peak,
                }])
                curve_rows.append({'Distance': d, 'Fare': float(model.predict(row)[0])})
            curve_df = pd.DataFrame(curve_rows)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_df['Distance'], y=curve_df['Fare'],
                mode='lines',
                line=dict(color=TEXT, width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=[distance], y=[predicted_fare],
                mode='markers',
                marker=dict(color=TEXT, size=8, line=dict(color=BG, width=2)),
                showlegend=False,
            ))
            apply_layout(fig, "",
                         xaxis_title="Distance (km)",
                         yaxis_title="Fare (Rs.)",
                         showlegend=False, height=CH,
                         margin=dict(t=10, b=30, l=45, r=15))
            st.plotly_chart(fig, use_container_width=True)

        # ── Vehicle comparison chart below ───────────────────────────────────
        compare_rows = []
        for vt in VEHICLE_TYPES:
            vt_code = int(le_vehicle.transform([vt])[0])
            row = pd.DataFrame([{
                'Ride_Distance':    distance,
                'Vehicle_Type_Enc': vt_code,
                'Hour':             hour,
                'DayOfWeek':        dow,
                'IsWeekend':        is_weekend,
                'IsNight':          is_night,
                'IsPeakHour':       is_peak,
            }])
            compare_rows.append({
                'Vehicle': vt,
                'Fare':    float(model.predict(row)[0]),
            })
        cmp_df = pd.DataFrame(compare_rows).sort_values('Fare').reset_index(drop=True)

        # Highlight the selected vehicle, everything else stays muted
        bar_colors = [TEXT if v == vehicle_type else BORDER for v in cmp_df['Vehicle']]
        text_colors = [TEXT if v == vehicle_type else MUTED for v in cmp_df['Vehicle']]

        st.markdown(f"""
        <div style="font-size:11px; color:{MUTED}; margin:18px 0 4px 0;">
            Same ride across all vehicle types
        </div>
        """, unsafe_allow_html=True)

        fig_cmp = go.Figure(go.Bar(
            x=cmp_df['Fare'], y=cmp_df['Vehicle'], orientation='h',
            marker=dict(color=bar_colors),
            text=[f"Rs. {f:,.0f}" for f in cmp_df['Fare']],
            textposition='outside',
            textfont=dict(size=10, color=text_colors),
            hovertemplate='%{y}: Rs. %{x:,.0f}<extra></extra>',
        ))
        apply_layout(fig_cmp, "",
                     showlegend=False, height=230,
                     margin=dict(t=10, b=25, l=90, r=60),
                     xaxis_title="Fare (Rs.)")
        # Make room for the outside text labels
        fig_cmp.update_xaxes(
            range=[0, cmp_df['Fare'].max() * 1.18],
            showgrid=True, gridcolor=BORDER,
        )
        fig_cmp.update_yaxes(showgrid=False)
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.markdown(f"""
        <div style="color:{MUTED}; font-size:13px; padding:24px 0;">
            Enter ride details above and click Calculate fare.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: INSIGHTS — dataset summary + model performance
# ═════════════════════════════════════════════════════════════════════════════
with tab_insights:
    # ── KPI metrics: 3 dataset stats + 3 model stats ──────────────────────────
    total_rides = len(df)
    avg_fare    = df[df['Booking_Status'] == 'Success']['Booking_Value'].mean()
    avg_dist    = df[df['Booking_Status'] == 'Success']['Ride_Distance'].mean()

    y_actual    = pred['Actual_Fare']
    y_predicted = pred['Predicted_Fare']
    mae  = (y_actual - y_predicted).abs().mean()
    rmse = ((y_actual - y_predicted) ** 2).mean() ** 0.5
    r2   = 1 - ((y_actual - y_predicted) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total rides",  f"{total_rides:,}")
    m2.metric("Avg fare",     f"Rs. {avg_fare:.0f}")
    m3.metric("Avg distance", f"{avg_dist:.1f} km")
    m4.metric("Model R\u00b2", f"{r2:.3f}")
    m5.metric("MAE",          f"Rs. {mae:.1f}")
    m6.metric("RMSE",         f"Rs. {rmse:.1f}")

    # ── Row 1: dataset charts ─────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        # Booking status distribution (real dataset)
        status_counts = df['Booking_Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = go.Figure(go.Pie(
            labels=status_counts['Status'], values=status_counts['Count'],
            hole=0.5,
            marker=dict(colors=[STATUS_COLORS.get(s, MUTED) for s in status_counts['Status']]),
            textinfo='percent',
            textfont=dict(size=10, color=TEXT),
        ))
        apply_layout(fig, "Booking status (real data)",
                     legend=dict(font=dict(size=8), orientation='h', y=-0.08))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Fare by vehicle type (real dataset)
        df_success = df[df['Booking_Status'] == 'Success']
        vehicle_palette = [PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT]
        fig2 = go.Figure()
        for i, vt in enumerate(sorted(df_success['Vehicle_Type'].dropna().unique())):
            fig2.add_trace(go.Box(
                y=df_success[df_success['Vehicle_Type'] == vt]['Booking_Value'],
                name=vt,
                marker=dict(color=vehicle_palette[i % len(vehicle_palette)]),
                boxpoints=False,
            ))
        apply_layout(fig2, "Fare by vehicle (real data)",
                     showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Bookings by hour (real dataset)
        hourly = df.groupby('Hour').size().reset_index(name='Bookings')
        fig3 = go.Figure(go.Bar(
            x=hourly['Hour'], y=hourly['Bookings'],
            marker=dict(color=PRIMARY),
        ))
        apply_layout(fig3, "Bookings by hour (real data)",
                     xaxis=dict(dtick=3), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Row 2: model performance charts ───────────────────────────────────────
    c4, c5, c6 = st.columns(3)

    with c4:
        # Actual vs predicted fare (model)
        sample = pred.sample(min(2500, len(pred)), random_state=42)
        fig4 = go.Figure()
        for i, vt in enumerate(sorted(sample['Vehicle_Type'].unique())):
            vt_rows = sample[sample['Vehicle_Type'] == vt]
            fig4.add_trace(go.Scatter(
                x=vt_rows['Actual_Fare'], y=vt_rows['Predicted_Fare'],
                mode='markers', name=vt,
                marker=dict(size=3, color=vehicle_palette[i % len(vehicle_palette)], opacity=0.5),
            ))
        fig4.add_trace(go.Scatter(
            x=[0, y_actual.max()], y=[0, y_actual.max()],
            mode='lines', line=dict(color=MUTED, dash='dash', width=1),
            showlegend=False, hoverinfo='skip',
        ))
        apply_layout(fig4, "Actual vs predicted fare",
                     legend=dict(font=dict(size=7), orientation='h', y=-0.18))
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        # Feature importances (model)
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig5 = go.Figure(go.Bar(
            x=fi.values, y=fi.index, orientation='h',
            marker=dict(color=GREEN),
            text=[f"{v:.3f}" for v in fi.values],
            textposition='outside',
            textfont=dict(size=9, color=TEXT),
        ))
        apply_layout(fig5, "Feature importances", showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        # Residual distribution (model)
        residuals = y_actual - y_predicted
        fig6 = go.Figure(go.Histogram(
            x=residuals, nbinsx=50,
            marker=dict(color=PRIMARY),
        ))
        apply_layout(fig6, "Prediction error distribution",
                     xaxis_title="Error (Rs.)", showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
