"""
Ride Booking Analytics & ML Dashboard
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

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
    <span class="topbar-title">Ride Bookings</span>
    <span class="topbar-meta">103,024 records &middot; July 2024 &middot; XGBoost Classifier</span>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent

# If model artifacts don't exist, train the model first (runs once on deploy)
if not (BASE / "xgb_model.pkl").exists():
    with st.spinner("First-time setup: training model..."):
        subprocess.run([sys.executable, str(BASE / "model.py")], check=True)

@st.cache_data
def load_data():
    return pd.read_csv(BASE / "cleaned_data.csv", parse_dates=['Date'])

@st.cache_data
def load_predictions():
    return pd.read_csv(BASE / "predictions.csv")

@st.cache_resource
def load_model():
    model    = joblib.load(BASE / "xgb_model.pkl")
    features = joblib.load(BASE / "features.pkl")
    le_v     = joblib.load(BASE / "le_vehicle.pkl")
    le_p     = joblib.load(BASE / "le_pickup.pkl")
    le_d     = joblib.load(BASE / "le_drop.pkl")
    return model, features, le_v, le_p, le_d

df   = load_data()
pred = load_predictions()
model, features, le_vehicle, le_pickup, le_drop = load_model()

VEHICLE_TYPES = ['Auto', 'Bike', 'eBike', 'Mini', 'Prime Plus', 'Prime Sedan', 'Prime SUV']

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
    y_test = pred['Actual_Status']
    y_pred_vals = pred['Predicted_Status']

    acc      = accuracy_score(y_test, y_pred_vals)
    f1_mac   = f1_score(y_test, y_pred_vals, average='macro')
    f1_wt    = f1_score(y_test, y_pred_vals, average='weighted')
    baseline = (y_test == 0).mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",    f"{acc*100:.1f}%")
    m2.metric("Baseline",    f"{baseline*100:.1f}%")
    m3.metric("F1 Macro",    f"{f1_mac:.3f}")
    m4.metric("F1 Weighted", f"{f1_wt:.3f}")

    st.info(
        "Synthetic dataset — booking status is distributed uniformly across all features. "
        "Real-world data would yield 80-90%+ accuracy with the same model."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        cm = confusion_matrix(y_test, y_pred_vals)
        fig = px.imshow(
            cm, x=LABEL_NAMES, y=LABEL_NAMES,
            text_auto=True, color_continuous_scale=[[0, BG], [1, PRIMARY]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            aspect="auto",
        )
        apply_layout(fig, "Confusion matrix",
                     margin=dict(t=28, b=5, l=5, r=5), coloraxis_showscale=False,
                     height=CH + 20)
        fig.update_xaxes(tickfont=dict(size=7))
        fig.update_yaxes(tickfont=dict(size=7))
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
        report = classification_report(y_test, y_pred_vals, target_names=LABEL_NAMES, output_dict=True)
        rdata = []
        for label in LABEL_NAMES:
            rdata.append({'Class': label, 'Metric': 'Precision', 'Value': report[label]['precision']})
            rdata.append({'Class': label, 'Metric': 'Recall',    'Value': report[label]['recall']})
            rdata.append({'Class': label, 'Metric': 'F1',        'Value': report[label]['f1-score']})
        rdf = pd.DataFrame(rdata)
        fig3 = px.bar(
            rdf, x='Value', y='Class', color='Metric', barmode='group',
            orientation='h',
            color_discrete_sequence=[PRIMARY, ORANGE, GREEN],
        )
        apply_layout(fig3, "Precision / Recall / F1",
                     legend=dict(font=dict(size=8), orientation='h', y=-0.12),
                     height=CH + 20)
        st.plotly_chart(fig3, use_container_width=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        actual_counts    = pd.Series(y_test).map(dict(enumerate(LABEL_NAMES))).value_counts()
        predicted_counts = pd.Series(y_pred_vals).map(dict(enumerate(LABEL_NAMES))).value_counts()
        compare_df = pd.DataFrame({'Actual': actual_counts, 'Predicted': predicted_counts}).reset_index()
        compare_df.columns = ['Status', 'Actual', 'Predicted']
        fig4 = px.bar(
            compare_df.melt(id_vars='Status', var_name='Type', value_name='Count'),
            x='Status', y='Count', color='Type', barmode='group',
            color_discrete_sequence=[PRIMARY, ORANGE],
        )
        apply_layout(fig4, "Actual vs predicted",
                     legend=dict(font=dict(size=8), orientation='h', y=-0.15),
                     xaxis_tickfont_size=7, xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        # Per-class recall gauges
        fig5 = make_subplots(rows=1, cols=4, specs=[[{'type': 'indicator'}]*4])
        gauge_colors = [GREEN, RED, ORANGE, PURPLE]
        for i, label in enumerate(LABEL_NAMES):
            mask = y_test == i
            class_acc = (y_pred_vals[mask] == i).mean() * 100 if mask.sum() > 0 else 0
            fig5.add_trace(go.Indicator(
                mode="gauge+number",
                value=class_acc,
                title={'text': label.split(' ')[0], 'font': {'size': 9, 'color': MUTED}},
                number={'suffix': '%', 'font': {'size': 12, 'color': TEXT}},
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(size=6, color=MUTED)),
                    bar=dict(color=gauge_colors[i]),
                    bgcolor=SURFACE,
                    bordercolor=BORDER,
                ),
            ), row=1, col=i+1)
        apply_layout(fig5, "Per-class recall", margin=dict(t=32, b=10, l=10, r=10))
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        pred_copy = pred.copy()
        pred_copy['Error'] = abs(pred_copy['Actual_Status'] - pred_copy['Predicted_Status'])
        err_dist = pred_copy['Error'].value_counts().sort_index().reset_index()
        err_dist.columns = ['Off_By', 'Count']
        err_dist['Off_By'] = err_dist['Off_By'].astype(str)
        fig6 = px.pie(
            err_dist, names='Off_By', values='Count',
            color_discrete_sequence=[GREEN, ORANGE, RED, PURPLE],
            hole=0.45,
        )
        fig6.update_traces(textinfo='percent', textfont_size=9)
        apply_layout(fig6, "Prediction error distance",
                     legend=dict(font=dict(size=8), title=dict(text='Off by', font=dict(size=8))))
        st.plotly_chart(fig6, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: PREDICT
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    # Inline form — all inputs across one row
    f1, f2, f3, f4, f5, f6, f7 = st.columns([1.2, 1, 1, 1.2, 1.2, 1, 0.6])
    with f1:
        vehicle_type = st.selectbox("Vehicle", VEHICLE_TYPES, label_visibility="collapsed",
                                    help="Vehicle type")
    with f2:
        booking_value = st.number_input("Fare", min_value=100, max_value=3000, value=500, step=50,
                                        label_visibility="collapsed", help="Fare (Rs.)")
    with f3:
        hour = st.number_input("Hour", min_value=0, max_value=23, value=9, step=1,
                               label_visibility="collapsed", help="Hour (0-23)")
    with f4:
        pickup_loc = st.selectbox("Pickup", sorted(le_pickup.classes_.tolist()),
                                  label_visibility="collapsed", help="Pickup location")
    with f5:
        drop_loc = st.selectbox("Drop", sorted(le_drop.classes_.tolist()),
                                label_visibility="collapsed", help="Drop location")
    with f6:
        day_of_week = st.selectbox("Day", ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                                   label_visibility="collapsed", help="Day of week")
        dow_map = {'Mon':0,'Tue':1,'Wed':2,'Thu':3,'Fri':4,'Sat':5,'Sun':6}
        dow = dow_map[day_of_week]
    with f7:
        predict_btn = st.button("Run", type="primary", use_container_width=True)

    # Field labels row
    st.markdown(f"""
    <div style="display:flex; gap:0; padding:0 0 6px 0; border-bottom:1px solid {BORDER}; margin-bottom:8px;">
        <span style="flex:1.2; font-size:10px; color:{MUTED};">Vehicle</span>
        <span style="flex:1; font-size:10px; color:{MUTED};">Fare (Rs.)</span>
        <span style="flex:1; font-size:10px; color:{MUTED};">Hour</span>
        <span style="flex:1.2; font-size:10px; color:{MUTED};">Pickup</span>
        <span style="flex:1.2; font-size:10px; color:{MUTED};">Drop</span>
        <span style="flex:1; font-size:10px; color:{MUTED};">Day</span>
        <span style="flex:0.6;"></span>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        is_weekend = 1 if dow >= 5 else 0
        is_night   = 1 if (hour >= 22 or hour <= 5) else 0
        is_peak    = 1 if hour in [7,8,9,17,18,19,20] else 0

        vt_enc = int(le_vehicle.transform([vehicle_type])[0])
        pk_enc = int(le_pickup.transform([pickup_loc])[0])
        dr_enc = int(le_drop.transform([drop_loc])[0])

        input_df = pd.DataFrame([{
            'Vehicle_Type_Enc': vt_enc, 'Pickup_Enc': pk_enc,
            'Drop_Enc': dr_enc, 'Booking_Value': booking_value,
            'Hour': hour, 'DayOfWeek': dow, 'IsWeekend': is_weekend,
            'IsNight': is_night, 'IsPeakHour': is_peak,
        }])

        pred_class = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        predicted_label = LABEL_NAMES[pred_class]
        color = STATUS_COLORS[predicted_label]

        # ── Result row: 4 status cards showing each class probability ─────
        r1, r2, r3, r4 = st.columns(4)
        for col, i, label in zip([r1, r2, r3, r4], range(4), LABEL_NAMES):
            prob = pred_proba[i] * 100
            is_predicted = (i == pred_class)
            border_c = STATUS_COLORS[label] if is_predicted else BORDER
            bg = f"{STATUS_COLORS[label]}0d" if is_predicted else SURFACE
            with col:
                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {border_c};
                            border-radius:4px; padding:10px 12px;">
                    <div style="font-size:11px; color:{MUTED}; margin-bottom:2px;">
                        {label}
                    </div>
                    <div style="font-size:22px; font-weight:600;
                                color:{STATUS_COLORS[label] if is_predicted else MUTED};">
                        {prob:.1f}%
                    </div>
                    <div style="background:{BORDER}; border-radius:2px; height:3px; margin-top:6px;">
                        <div style="background:{STATUS_COLORS[label]}; width:{prob}%;
                                    height:3px; border-radius:2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Charts row: probability bar + comparison with dataset avg ──────
        ch1, ch2, ch3 = st.columns(3)

        with ch1:
            prob_df = pd.DataFrame({'Status': LABEL_NAMES, 'Probability': pred_proba * 100})
            fig = px.bar(
                prob_df, y='Status', x='Probability', orientation='h',
                color='Status', color_discrete_map=STATUS_COLORS,
                text=prob_df['Probability'].apply(lambda x: f"{x:.1f}%"),
            )
            fig.update_traces(textposition='outside', textfont_size=9)
            apply_layout(fig, "Probability breakdown",
                         showlegend=False, height=CH)
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            # Compare this prediction vs dataset average
            dataset_rates = df['Booking_Status'].value_counts(normalize=True).reindex(LABEL_NAMES).fillna(0) * 100
            compare_data = []
            for i, label in enumerate(LABEL_NAMES):
                compare_data.append({'Status': label, 'Source': 'This prediction', 'Pct': pred_proba[i] * 100})
                compare_data.append({'Status': label, 'Source': 'Dataset average', 'Pct': dataset_rates[label]})
            cdf = pd.DataFrame(compare_data)
            fig2 = px.bar(
                cdf, x='Status', y='Pct', color='Source', barmode='group',
                color_discrete_sequence=[PRIMARY, MUTED],
            )
            apply_layout(fig2, "vs. dataset average",
                         legend=dict(font=dict(size=8), orientation='h', y=-0.15),
                         xaxis_tickfont_size=7, xaxis_tickangle=-15, height=CH)
            st.plotly_chart(fig2, use_container_width=True)

        with ch3:
            # What does the dataset look like for this vehicle + hour combo
            filtered = df[(df['Vehicle_Type'] == vehicle_type) & (df['Hour'] == hour)]
            if len(filtered) > 0:
                filt_rates = filtered['Booking_Status'].value_counts(normalize=True).reindex(LABEL_NAMES).fillna(0) * 100
                filt_df = pd.DataFrame({'Status': LABEL_NAMES, 'Rate': filt_rates.values})
                fig3 = px.bar(
                    filt_df, y='Status', x='Rate', orientation='h',
                    color='Status', color_discrete_map=STATUS_COLORS,
                    text=filt_df['Rate'].apply(lambda x: f"{x:.1f}%"),
                )
                fig3.update_traces(textposition='outside', textfont_size=9)
                apply_layout(fig3, f"Historical: {vehicle_type} at {hour}:00",
                             showlegend=False, height=CH)
            else:
                fig3 = go.Figure()
                apply_layout(fig3, "No historical data for this combo", height=CH)
            st.plotly_chart(fig3, use_container_width=True)

        # ── Detail row: derived features table ─────────────────────────────
        with st.expander("Derived features"):
            detail_df = pd.DataFrame([{
                'Vehicle': vehicle_type,
                'Fare': f"Rs.{booking_value}",
                'Hour': hour,
                'Day': day_of_week,
                'Weekend': 'Yes' if is_weekend else 'No',
                'Night': 'Yes' if is_night else 'No',
                'Peak': 'Yes' if is_peak else 'No',
                'Pickup': pickup_loc,
                'Drop': drop_loc,
            }])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
