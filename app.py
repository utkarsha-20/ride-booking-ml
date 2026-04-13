"""
Ride Booking Analytics & ML Dashboard
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import plotly.express as px
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
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analytics", "Model", "Predict"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    total       = len(df)
    success     = (df['Booking_Status'] == 'Success').sum()
    canceled_d  = (df['Booking_Status'] == 'Canceled by Driver').sum()
    canceled_c  = (df['Booking_Status'] == 'Canceled by Customer').sum()
    not_found   = (df['Booking_Status'] == 'Driver Not Found').sum()
    avg_fare    = df[df['Booking_Status'] == 'Success']['Booking_Value'].mean()
    avg_dist    = df[df['Booking_Status'] == 'Success']['Ride_Distance'].mean()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total",       f"{total:,}")
    m2.metric("Success",     f"{success:,}")
    m3.metric("Canc. (Drv)", f"{canceled_d:,}")
    m4.metric("Canc. (Cust)",f"{canceled_c:,}")
    m5.metric("Avg Fare",    f"Rs.{avg_fare:.0f}")
    m6.metric("Avg Distance",f"{avg_dist:.1f} km")

    c1, c2, c3 = st.columns(3)

    with c1:
        status_counts = df['Booking_Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.pie(
            status_counts, names='Status', values='Count',
            color='Status', color_discrete_map=STATUS_COLORS, hole=0.5,
        )
        fig.update_traces(textinfo='percent', textfont_size=9)
        apply_layout(fig, "Status distribution",
                     legend=dict(font=dict(size=8), orientation='h', y=-0.08))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        vt_status = df.groupby(['Vehicle_Type', 'Booking_Status']).size().reset_index(name='Count')
        fig2 = px.bar(
            vt_status, x='Vehicle_Type', y='Count', color='Booking_Status',
            color_discrete_map=STATUS_COLORS, barmode='stack',
        )
        apply_layout(fig2, "Bookings by vehicle",
                     legend=dict(font=dict(size=8), orientation='h', y=-0.12),
                     xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        succ_rate = (
            df.groupby('Vehicle_Type')['Booking_Status']
            .apply(lambda x: (x == 'Success').mean() * 100)
            .reset_index()
        )
        succ_rate.columns = ['Vehicle_Type', 'Rate']
        succ_rate = succ_rate.sort_values('Rate', ascending=True)
        fig3 = px.bar(
            succ_rate, x='Rate', y='Vehicle_Type', orientation='h',
            color='Rate', color_continuous_scale=[[0, RED], [0.5, ORANGE], [1, GREEN]],
            text=succ_rate['Rate'].apply(lambda x: f"{x:.1f}%"),
            range_x=[58, 66],
        )
        fig3.update_traces(textposition='outside', textfont_size=9)
        apply_layout(fig3, "Success rate", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        hourly = df.groupby('Hour')['Booking_Status'].apply(
            lambda x: (x == 'Success').mean() * 100
        ).reset_index()
        hourly.columns = ['Hour', 'Rate']
        fig4 = px.area(hourly, x='Hour', y='Rate', color_discrete_sequence=[PRIMARY])
        apply_layout(fig4, "Success rate by hour", xaxis=dict(dtick=4))
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        daily = df.groupby('DayOfWeek').size().reset_index(name='Bookings')
        daily['Day'] = daily['DayOfWeek'].map(day_map)
        fig5 = px.bar(
            daily, x='Day', y='Bookings', color='Bookings',
            color_continuous_scale=[[0, SURFACE], [1, PRIMARY]],
            category_orders={'Day': list(day_map.values())},
        )
        apply_layout(fig5, "Bookings by day", coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        df_s = df[df['Booking_Status'] == 'Success']
        df_s = df_s[df_s['Ride_Distance'] > 0].copy()
        df_s['Dist_Bin'] = pd.cut(
            df_s['Ride_Distance'],
            bins=[0, 10, 20, 30, 50], labels=['0-10km', '10-20km', '20-30km', '30-50km']
        )
        dist_vt = df_s.groupby(['Dist_Bin', 'Vehicle_Type'], observed=True).size().reset_index(name='Count')
        fig6 = px.sunburst(
            dist_vt, path=['Dist_Bin', 'Vehicle_Type'], values='Count',
            color='Count', color_continuous_scale=[[0, SURFACE], [1, PRIMARY]],
        )
        apply_layout(fig6, "Distance x vehicle",
                     margin=dict(t=28, b=5, l=5, r=5), coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    df_success = df[df['Booking_Status'] == 'Success']

    c1, c2, c3 = st.columns(3)

    with c1:
        hourly = df.groupby(['Hour', 'Booking_Status']).size().reset_index(name='Count')
        fig = px.bar(
            hourly, x='Hour', y='Count', color='Booking_Status',
            color_discrete_map=STATUS_COLORS, barmode='stack',
        )
        apply_layout(fig, "Hourly pattern",
                     legend=dict(font=dict(size=7), orientation='h', y=-0.15),
                     xaxis=dict(dtick=4))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.box(
            df_success, x='Vehicle_Type', y='Booking_Value',
            color='Vehicle_Type',
            color_discrete_sequence=[PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT],
            points=False,
        )
        apply_layout(fig2, "Fare by vehicle", showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        sample = df_success.sample(min(2500, len(df_success)), random_state=42)
        fig3 = px.scatter(
            sample, x='Ride_Distance', y='Booking_Value',
            color='Vehicle_Type', opacity=0.4,
            color_discrete_sequence=[PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT],
        )
        fig3.update_traces(marker_size=2)
        apply_layout(fig3, "Distance vs fare",
                     legend=dict(font=dict(size=7), orientation='h', y=-0.15))
        st.plotly_chart(fig3, use_container_width=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        fig4 = px.histogram(
            df_success, x='Booking_Value', nbins=50,
            color_discrete_sequence=[PRIMARY],
        )
        apply_layout(fig4, "Fare distribution")
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        df_dist = df_success[df_success['Ride_Distance'] > 0]
        fig5 = px.histogram(
            df_dist, x='Ride_Distance', nbins=40,
            color_discrete_sequence=[PURPLE],
        )
        apply_layout(fig5, "Distance distribution")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        vt_stats = df_success.groupby('Vehicle_Type').agg(
            Fare=('Booking_Value', 'mean'),
            Dist=('Ride_Distance', 'mean'),
            DrvR=('Driver_Ratings', 'mean'),
            CustR=('Customer_Rating', 'mean'),
        ).reset_index()
        for c in ['Fare', 'Dist', 'DrvR', 'CustR']:
            vt_stats[c] = (vt_stats[c] - vt_stats[c].min()) / (vt_stats[c].max() - vt_stats[c].min() + 1e-9)
        fig6 = go.Figure()
        cats = ['Fare', 'Distance', 'Driver Rating', 'Customer Rating']
        colors = [PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT]
        for i, (_, row) in enumerate(vt_stats.iterrows()):
            vals = [row['Fare'], row['Dist'], row['DrvR'], row['CustR']]
            fig6.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                name=row['Vehicle_Type'], fill='toself', opacity=0.4,
                line=dict(color=colors[i % len(colors)]),
            ))
        apply_layout(fig6, "Vehicle comparison",
                     polar=dict(
                         radialaxis=dict(visible=False),
                         bgcolor='rgba(0,0,0,0)',
                     ),
                     legend=dict(font=dict(size=7), orientation='h', y=-0.2),
                     margin=dict(t=28, b=20, l=50, r=50))
        st.plotly_chart(fig6, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    y_actual = pred['Actual_Fare']
    y_pred_vals = pred['Predicted_Fare']

    mae  = (y_actual - y_pred_vals).abs().mean()
    rmse = ((y_actual - y_pred_vals) ** 2).mean() ** 0.5
    r2   = 1 - ((y_actual - y_pred_vals) ** 2).sum() / ((y_actual - y_actual.mean()) ** 2).sum()
    mean_fare = y_actual.mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R squared",    f"{r2:.3f}")
    m2.metric("MAE",          f"Rs. {mae:.1f}")
    m3.metric("RMSE",         f"Rs. {rmse:.1f}")
    m4.metric("Mean fare",    f"Rs. {mean_fare:.0f}")

    st.info(
        "Model trained on simulated ride fares generated from a realistic pricing formula "
        "(base fare + per-km rate + surge pricing). The real Bookings.xlsx dataset is used "
        "throughout the Overview and Analytics tabs."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        # Actual vs predicted scatter
        sample = pred.sample(min(3000, len(pred)), random_state=42)
        fig = px.scatter(
            sample, x='Actual_Fare', y='Predicted_Fare',
            color='Vehicle_Type', opacity=0.5,
            color_discrete_sequence=[PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT],
        )
        fig.add_trace(go.Scatter(
            x=[0, y_actual.max()], y=[0, y_actual.max()],
            mode='lines', line=dict(color=MUTED, dash='dash', width=1),
            showlegend=False, hoverinfo='skip',
        ))
        fig.update_traces(marker_size=3, selector=dict(mode='markers'))
        apply_layout(fig, "Actual vs predicted fare",
                     legend=dict(font=dict(size=7), orientation='h', y=-0.18),
                     height=CH + 20)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig2 = px.bar(
            fi.reset_index(), x=0, y='index', orientation='h',
            color=0, color_continuous_scale=[[0, SURFACE], [1, GREEN]],
            labels={'index': '', 0: ''},
        )
        apply_layout(fig2, "Feature importances",
                     coloraxis_showscale=False, height=CH + 20)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Residual (error) distribution
        residuals = y_actual - y_pred_vals
        fig3 = px.histogram(
            residuals, nbins=50,
            color_discrete_sequence=[PRIMARY],
        )
        apply_layout(fig3, "Residual distribution",
                     xaxis_title="Error (Rs.)", height=CH + 20, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        # Error by vehicle type
        pred_copy = pred.copy()
        pred_copy['Abs_Error'] = (pred_copy['Actual_Fare'] - pred_copy['Predicted_Fare']).abs()
        err_by_vehicle = pred_copy.groupby('Vehicle_Type')['Abs_Error'].mean().reset_index()
        err_by_vehicle = err_by_vehicle.sort_values('Abs_Error', ascending=True)
        fig4 = px.bar(
            err_by_vehicle, x='Abs_Error', y='Vehicle_Type',
            orientation='h',
            color='Abs_Error',
            color_continuous_scale=[[0, GREEN], [1, ORANGE]],
            text=err_by_vehicle['Abs_Error'].apply(lambda x: f"Rs.{x:.0f}"),
        )
        fig4.update_traces(textposition='outside', textfont_size=9)
        apply_layout(fig4, "MAE by vehicle type",
                     coloraxis_showscale=False, height=CH)
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        # Predicted fare by distance (line chart showing the model's learned relationship)
        line_df = pred.copy()
        line_df['Dist_Bin'] = pd.cut(line_df['Ride_Distance'], bins=20)
        line_agg = line_df.groupby(['Dist_Bin', 'Vehicle_Type'], observed=True)['Predicted_Fare'].mean().reset_index()
        line_agg['Dist_Mid'] = line_agg['Dist_Bin'].apply(lambda x: x.mid)
        fig5 = px.line(
            line_agg, x='Dist_Mid', y='Predicted_Fare', color='Vehicle_Type',
            color_discrete_sequence=[PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT],
        )
        apply_layout(fig5, "Predicted fare vs distance",
                     xaxis_title="Distance (km)", yaxis_title="Fare (Rs.)",
                     legend=dict(font=dict(size=7), orientation='h', y=-0.18),
                     height=CH)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        # Percentage error distribution
        pct_err = ((y_actual - y_pred_vals) / y_actual * 100).abs()
        bins_data = pd.cut(pct_err, bins=[0, 5, 10, 20, 50, 100]).value_counts().sort_index()
        bin_df = pd.DataFrame({
            'Range': ['<5%', '5-10%', '10-20%', '20-50%', '>50%'],
            'Count': bins_data.values,
        })
        fig6 = px.pie(
            bin_df, names='Range', values='Count',
            color_discrete_sequence=[GREEN, PRIMARY, ORANGE, RED, PURPLE],
            hole=0.45,
        )
        fig6.update_traces(textinfo='percent', textfont_size=9)
        apply_layout(fig6, "Error % buckets",
                     legend=dict(font=dict(size=8)))
        st.plotly_chart(fig6, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: PREDICT — Fare calculator
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    # Inline form — all inputs across one row
    f1, f2, f3, f4, f5 = st.columns([1.3, 1.3, 1, 1.3, 0.7])
    with f1:
        vehicle_type = st.selectbox("Vehicle", VEHICLE_TYPES, index=4,
                                    label_visibility="collapsed")
    with f2:
        distance = st.slider("Distance", min_value=1, max_value=50, value=15, step=1,
                             label_visibility="collapsed")
    with f3:
        hour = st.slider("Hour", min_value=0, max_value=23, value=9, step=1,
                         label_visibility="collapsed")
    with f4:
        day_of_week = st.selectbox("Day",
                                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                                    label_visibility="collapsed")
        dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}
        dow = dow_map[day_of_week]
    with f5:
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    # Field labels row
    st.markdown(f"""
    <div style="display:flex; gap:0; padding:0 0 6px 0; border-bottom:1px solid {BORDER}; margin-bottom:10px;">
        <span style="flex:1.3; font-size:10px; color:{MUTED};">Vehicle type</span>
        <span style="flex:1.3; font-size:10px; color:{MUTED};">Distance (km)</span>
        <span style="flex:1; font-size:10px; color:{MUTED};">Hour</span>
        <span style="flex:1.3; font-size:10px; color:{MUTED};">Day</span>
        <span style="flex:0.7;"></span>
    </div>
    """, unsafe_allow_html=True)

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

        # Determine active surge tags
        surge_tags = []
        if is_peak:    surge_tags.append("Peak hour +25%")
        if is_night:   surge_tags.append("Late night +30%")
        if is_weekend: surge_tags.append("Weekend +10%")
        surge_html = " &middot; ".join(surge_tags) if surge_tags else "No surge"

        # ── Result banner: big fare number ─────────────────────────────────
        st.markdown(f"""
        <div style="background:{SURFACE}; border:1px solid {BORDER};
                    border-radius:6px; padding:16px 20px; margin-bottom:10px;">
            <div style="display:flex; align-items:baseline; gap:16px;">
                <div>
                    <div style="font-size:11px; color:{MUTED}; margin-bottom:2px;">
                        Predicted fare
                    </div>
                    <div style="font-size:32px; font-weight:600; color:{GREEN}; line-height:1;">
                        Rs. {predicted_fare:,.0f}
                    </div>
                </div>
                <div style="margin-left:auto; text-align:right;">
                    <div style="font-size:11px; color:{MUTED}; margin-bottom:2px;">
                        {distance} km &middot; {vehicle_type} &middot; {hour:02d}:00
                    </div>
                    <div style="font-size:11px; color:{MUTED};">
                        {surge_html}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts row ─────────────────────────────────────────────────────
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            # Compare fare across all vehicle types for this distance & time
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
                    'Fare': float(model.predict(row)[0])
                })
            cmp_df = pd.DataFrame(compare_rows).sort_values('Fare')
            colors = [GREEN if v == vehicle_type else MUTED for v in cmp_df['Vehicle']]
            fig = go.Figure(go.Bar(
                x=cmp_df['Fare'], y=cmp_df['Vehicle'], orientation='h',
                text=[f"Rs.{f:.0f}" for f in cmp_df['Fare']],
                textposition='outside',
                textfont=dict(size=9, color=TEXT),
                marker=dict(color=colors),
            ))
            apply_layout(fig, "Fare across vehicles",
                         showlegend=False, height=CH)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            # Fare curve across all distances for selected vehicle
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
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=curve_df['Distance'], y=curve_df['Fare'],
                mode='lines', line=dict(color=PRIMARY, width=2),
                fill='tozeroy', fillcolor=f"{PRIMARY}22",
            ))
            # Mark selected distance
            fig2.add_trace(go.Scatter(
                x=[distance], y=[predicted_fare],
                mode='markers', marker=dict(color=GREEN, size=10, line=dict(color=TEXT, width=1)),
            ))
            apply_layout(fig2, f"Fare curve ({vehicle_type})",
                         xaxis_title="Distance (km)", yaxis_title="Fare (Rs.)",
                         showlegend=False, height=CH)
            st.plotly_chart(fig2, use_container_width=True)

        with ch3:
            # Fare across all hours for selected vehicle + distance
            hour_rows = []
            for h in range(24):
                is_n = 1 if (h >= 22 or h <= 4) else 0
                is_p = 1 if h in (7, 8, 9, 17, 18, 19, 20) else 0
                row = pd.DataFrame([{
                    'Ride_Distance':    distance,
                    'Vehicle_Type_Enc': vt_enc,
                    'Hour':             h,
                    'DayOfWeek':        dow,
                    'IsWeekend':        is_weekend,
                    'IsNight':          is_n,
                    'IsPeakHour':       is_p,
                }])
                hour_rows.append({'Hour': h, 'Fare': float(model.predict(row)[0])})
            hour_df = pd.DataFrame(hour_rows)
            bar_colors = [GREEN if h == hour else PRIMARY for h in hour_df['Hour']]
            fig3 = go.Figure(go.Bar(
                x=hour_df['Hour'], y=hour_df['Fare'],
                marker=dict(color=bar_colors),
            ))
            apply_layout(fig3, f"Fare by hour ({distance} km, {vehicle_type})",
                         xaxis_title="Hour", yaxis_title="Fare (Rs.)",
                         showlegend=False, height=CH,
                         xaxis=dict(dtick=3))
            st.plotly_chart(fig3, use_container_width=True)

        # ── Details ────────────────────────────────────────────────────────
        with st.expander("Model inputs & breakdown"):
            detail_df = pd.DataFrame([{
                'Vehicle':       vehicle_type,
                'Distance (km)': distance,
                'Hour':          f"{hour:02d}:00",
                'Day':           day_of_week,
                'Weekend':       'Yes' if is_weekend else 'No',
                'Night surge':   'Yes' if is_night else 'No',
                'Peak surge':    'Yes' if is_peak else 'No',
                'Predicted':     f"Rs. {predicted_fare:,.0f}",
            }])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
    else:
        # Default empty state message
        st.markdown(f"""
        <div style="padding:40px 20px; text-align:center; color:{MUTED}; font-size:13px;
                    border:1px dashed {BORDER}; border-radius:6px;">
            Choose a vehicle, distance, hour, and day &mdash; then click <b>Predict</b>.
        </div>
        """, unsafe_allow_html=True)
