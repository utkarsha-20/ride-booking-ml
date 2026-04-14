"""
Ride Booking Analytics & ML Dashboard
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

import db

st.set_page_config(
    page_title="FareCast",
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
        padding: 3rem 1.2rem 0.4rem 1.2rem !important;
        max-width: 100% !important;
    }}

    /* Top bar */
    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid {BORDER};
        padding: 0 0 10px 0;
        margin-bottom: 10px;
    }}
    .topbar-title {{
        font-size: 32px;
        font-weight: 700;
        color: #ff9e64;
        letter-spacing: -0.8px;
        line-height: 1;
    }}
    .topbar-tag {{
        font-size: 13px;
        color: {MUTED};
        margin-left: 12px;
        font-weight: 400;
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

    /* Footer byline — fixed bottom-right, unobtrusive */
    .byline {{
        position: fixed;
        bottom: 10px;
        right: 18px;
        font-size: 11px;
        color: {MUTED};
        z-index: 100;
        pointer-events: none;
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
    <div>
        <span class="topbar-title">FareCast</span>
        <span class="topbar-tag">ride fare prediction</span>
    </div>
    <span class="topbar-meta">103,024 records &middot; July 2024 &middot; XGBoost Regressor</span>
</div>
<div class="byline">Built by Utkarsha Bhad</div>
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

def _file_key(path):
    """File-size-based cache key. Changes whenever the file changes."""
    return path.stat().st_size if path.exists() else 0

@st.cache_data
def load_data(_cache_key):
    return pd.read_csv(BASE / "cleaned_data.csv", parse_dates=['Date'])

@st.cache_data
def load_predictions(_cache_key):
    return pd.read_csv(BASE / "predictions.csv")

@st.cache_resource
def load_model(_cache_key):
    model    = joblib.load(BASE / "xgb_model.pkl")
    features = joblib.load(BASE / "features.pkl")
    le_v     = joblib.load(BASE / "le_vehicle.pkl")
    return model, features, le_v

df   = load_data(_file_key(BASE / "cleaned_data.csv"))
pred = load_predictions(_file_key(BASE / "predictions.csv"))
model, features, le_vehicle = load_model(_file_key(BASE / "xgb_model.pkl"))

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

    # Per-vehicle color palette — each vehicle gets its own identity.
    # Twilight Mist inspired: soft purples, warm orange, dusty pink, teal, gold
    VEHICLE_COLORS = {
        'Bike':        '#7aa2f7',   # soft blue
        'eBike':       '#9ece6a',   # sage green
        'Auto':        '#ff9e64',   # warm orange
        'Mini':        '#e0af68',   # dusty gold
        'Prime Sedan': '#bb9af7',   # lavender
        'Prime Plus':  '#f7768e',   # dusty pink
        'Prime SUV':   '#9d7cd8',   # deeper purple
    }
    SURGE_COLORS = {
        'peak hour':  '#ff9e64',   # warm orange
        'late night': '#bb9af7',   # lavender
        'weekend':    '#9ece6a',   # sage green
    }

    # When the user clicks "Calculate fare", compute and store in session state
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

        st.session_state['last_prediction'] = {
            'vehicle_type':   vehicle_type,
            'distance':       int(distance),
            'hour':           hour,
            'hour_label':     hour_label,
            'day_of_week':    day_of_week,
            'dow':            dow,
            'is_weekend':     is_weekend,
            'is_night':       is_night,
            'is_peak':        is_peak,
            'vt_enc':         vt_enc,
            'predicted_fare': float(model.predict(input_df)[0]),
        }
        # A fresh prediction resets any save notice
        st.session_state.pop('save_notice', None)

    # Render results only if we have a stored prediction
    if 'last_prediction' in st.session_state:
        p = st.session_state['last_prediction']
        vehicle_type   = p['vehicle_type']
        distance       = p['distance']
        hour           = p['hour']
        hour_label     = p['hour_label']
        day_of_week    = p['day_of_week']
        dow            = p['dow']
        is_weekend     = p['is_weekend']
        is_night       = p['is_night']
        is_peak        = p['is_peak']
        vt_enc         = p['vt_enc']
        predicted_fare = p['predicted_fare']

        accent = VEHICLE_COLORS.get(vehicle_type, PURPLE)

        # Left side: result.  Right side: fare curve chart.
        left, right = st.columns([1, 1.6])

        with left:
            # Surge badges (only shown for active ones)
            surge_badges_html = ""
            active_surges = []
            if is_peak:    active_surges.append('peak hour')
            if is_night:   active_surges.append('late night')
            if is_weekend: active_surges.append('weekend')

            if active_surges:
                badges = []
                for s in active_surges:
                    c = SURGE_COLORS[s]
                    badges.append(
                        f'<span style="display:inline-block; font-size:11px; '
                        f'color:{c}; border:1px solid {c}; border-radius:4px; '
                        f'padding:2px 8px; margin-right:6px;">{s}</span>'
                    )
                surge_badges_html = "".join(badges)
            else:
                surge_badges_html = f'<span style="font-size:12px; color:{MUTED};">no surge</span>'

            st.markdown(f"""
            <div style="padding-bottom:12px; border-bottom:1px solid {BORDER};
                        margin-bottom:12px;">
                <div style="font-size:12px; color:{MUTED}; margin-bottom:4px;">Fare</div>
                <div style="font-size:32px; font-weight:600; color:{accent}; line-height:1;">
                    Rs. {predicted_fare:,.0f}
                </div>
            </div>

            <table style="width:100%; border-collapse:collapse; font-size:13px; color:{TEXT};">
                <tr>
                    <td style="padding:6px 0; color:{MUTED}; width:42%;">Vehicle</td>
                    <td style="padding:6px 0; color:{accent}; font-weight:500;">
                        {vehicle_type}
                    </td>
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
                    <td style="padding:10px 0 0 0; color:{MUTED};
                               border-top:1px solid {BORDER}; vertical-align:top;">Surge</td>
                    <td style="padding:10px 0 0 0; border-top:1px solid {BORDER};">
                        {surge_badges_html}
                    </td>
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

            # Convert the accent hex to rgba for the fill
            h = accent.lstrip('#')
            r_, g_, b_ = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            accent_fill = f'rgba({r_}, {g_}, {b_}, 0.12)'

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_df['Distance'], y=curve_df['Fare'],
                mode='lines',
                line=dict(color=accent, width=2.5),
                fill='tozeroy',
                fillcolor=accent_fill,
            ))
            fig.add_trace(go.Scatter(
                x=[distance], y=[predicted_fare],
                mode='markers',
                marker=dict(color=accent, size=12, line=dict(color=BG, width=2)),
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

        # Selected vehicle at full opacity; others dimmed so the pick stands out
        bar_colors  = [VEHICLE_COLORS[v] for v in cmp_df['Vehicle']]
        bar_opacity = [1.0 if v == vehicle_type else 0.28 for v in cmp_df['Vehicle']]
        # Label next to the selected bar gets a small indicator; others stay muted
        bar_texts = [
            f"Rs. {f:,.0f}   ← selected" if v == vehicle_type else f"Rs. {f:,.0f}"
            for v, f in zip(cmp_df['Vehicle'], cmp_df['Fare'])
        ]
        text_colors = [
            VEHICLE_COLORS[v] if v == vehicle_type else MUTED
            for v in cmp_df['Vehicle']
        ]
        # Color the y-axis tick labels so the selected vehicle name is accented
        tick_colors = [
            VEHICLE_COLORS[v] if v == vehicle_type else MUTED
            for v in cmp_df['Vehicle']
        ]

        st.markdown(f"""
        <div style="font-size:11px; color:{MUTED}; margin:18px 0 4px 0;">
            Same ride across all vehicle types
        </div>
        """, unsafe_allow_html=True)

        fig_cmp = go.Figure(go.Bar(
            x=cmp_df['Fare'], y=cmp_df['Vehicle'], orientation='h',
            marker=dict(color=bar_colors, opacity=bar_opacity),
            text=bar_texts,
            textposition='outside',
            textfont=dict(size=11, color=text_colors),
            hovertemplate='%{y}: Rs. %{x:,.0f}<extra></extra>',
        ))
        apply_layout(fig_cmp, "",
                     showlegend=False, height=240,
                     margin=dict(t=10, b=25, l=110, r=90),
                     xaxis_title="Fare (Rs.)")
        # Room for outside text labels including the "← selected" suffix
        fig_cmp.update_xaxes(
            range=[0, cmp_df['Fare'].max() * 1.32],
            showgrid=True, gridcolor=BORDER,
        )
        # Per-tick label coloring: color the selected vehicle name in its accent
        fig_cmp.update_yaxes(
            showgrid=False,
            tickmode='array',
            tickvals=list(cmp_df['Vehicle']),
            ticktext=[
                f"<span style='color:{tick_colors[i]};'><b>{v}</b></span>"
                if v == vehicle_type
                else f"<span style='color:{tick_colors[i]};'>{v}</span>"
                for i, v in enumerate(cmp_df['Vehicle'])
            ],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Save to MySQL ─────────────────────────────────────────────────
        st.markdown(
            f"<hr style='border:none; border-top:1px solid {BORDER}; "
            f"margin:18px 0 10px 0;'/>",
            unsafe_allow_html=True,
        )
        sc1, sc2 = st.columns([1, 3])
        with sc1:
            save_btn = st.button("Save to database", use_container_width=True)
        with sc2:
            notice = st.session_state.get('save_notice')
            if notice:
                kind, msg = notice
                color = accent if kind == 'ok' else '#f7768e'
                st.markdown(
                    f"<div style='padding:7px 0; color:{color}; font-size:12px;'>"
                    f"{msg}</div>",
                    unsafe_allow_html=True,
                )

        if save_btn:
            try:
                db.ensure_schema()
                row_id = db.save_prediction(
                    vehicle_type=vehicle_type,
                    ride_distance=distance,
                    hour=hour,
                    day_of_week=day_of_week,
                    is_weekend=is_weekend,
                    is_night=is_night,
                    is_peak_hour=is_peak,
                    predicted_fare=predicted_fare,
                )
                st.session_state['save_notice'] = (
                    'ok',
                    f"Saved as row #{row_id} in fare_predictions.",
                )
            except Exception as e:
                st.session_state['save_notice'] = (
                    'err',
                    f"Could not save: {e}",
                )
            st.rerun()
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

    # Shared palette used by the row 1 & row 2 charts
    vehicle_palette = [PRIMARY, GREEN, ORANGE, PURPLE, RED, MUTED, TEXT]

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
        # Ride distance distribution (real dataset, successful rides only)
        dist_data = df.loc[
            (df['Booking_Status'] == 'Success') & (df['Ride_Distance'] > 0),
            'Ride_Distance'
        ]
        fig2 = go.Figure(go.Histogram(
            x=dist_data, nbinsx=40,
            marker=dict(color=PURPLE),
            hovertemplate='%{x} km: %{y:,} rides<extra></extra>',
        ))
        apply_layout(fig2, "Ride distance distribution (real data)",
                     xaxis_title="Distance (km)",
                     yaxis_title="Rides",
                     showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        # Cancellation reasons (real dataset) — strong variation: 6.5x ratio
        if ('Canceled_Rides_by_Driver' in df.columns
            and 'Canceled_Rides_by_Customer' in df.columns):

            driver_reasons = df['Canceled_Rides_by_Driver'].value_counts()
            cust_reasons   = df['Canceled_Rides_by_Customer'].value_counts()

            # Build a combined sorted list — driver reasons + customer reasons
            rows = []
            for reason, count in driver_reasons.items():
                rows.append({'Reason': reason, 'Count': int(count), 'Side': 'Driver'})
            for reason, count in cust_reasons.items():
                rows.append({'Reason': reason, 'Count': int(count), 'Side': 'Customer'})
            rdf = pd.DataFrame(rows).sort_values('Count', ascending=True)

            colors = [RED if s == 'Driver' else ORANGE for s in rdf['Side']]

            fig3 = go.Figure(go.Bar(
                x=rdf['Count'], y=rdf['Reason'],
                orientation='h',
                marker=dict(color=colors),
                text=[f"{c:,}" for c in rdf['Count']],
                textposition='outside',
                textfont=dict(size=9, color=TEXT),
                hovertemplate='%{y}: %{x:,} rides<extra>%{customdata}</extra>',
                customdata=rdf['Side'],
            ))
            apply_layout(fig3, "Why rides get canceled (real data)",
                         showlegend=False,
                         margin=dict(t=28, b=20, l=5, r=60))
            fig3.update_xaxes(
                range=[0, rdf['Count'].max() * 1.22],
                showgrid=True, gridcolor=BORDER,
            )
            fig3.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=8))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Cancellation reason data not available.")

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
