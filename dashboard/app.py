# ══════════════════════════════════════════════════════════════════
#  FMCG SALES FORECASTING DASHBOARD  |  app.py
#  Run: streamlit run dashboard/app.py  (from project root)
# ══════════════════════════════════════════════════════════════════

import os, sys, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = r"E:\Imarticus BNGLR\FMCG Sales Forecsting 24-25\Dataset.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts", "model_results")

def _resolve_data():
    candidates = [
        os.path.join(BASE_DIR, "data", "raw", "FMCG_Sales_2024_2025_v2.csv"),
        os.path.join(BASE_DIR, "Dataset.csv"),
        DATA_FILE,
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return DATA_FILE

DATA_PATH = _resolve_data()

def model_path(fname): return os.path.join(MODELS_DIR, fname)
def models_ready():    return os.path.exists(model_path("model_metrics.json"))

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(page_title="FMCG Sales Intelligence",
                   page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.5rem;padding-bottom:1rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F172A 0%,#1E293B 100%);border-right:1px solid #334155;}
[data-testid="stSidebar"] *{color:#CBD5E1 !important;}
.kpi-card{background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);border:1px solid #334155;border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;transition:border-color .2s;}
.kpi-card:hover{border-color:#475569;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.kpi-blue::before{background:#3B82F6;} .kpi-green::before{background:#10B981;}
.kpi-amber::before{background:#F59E0B;} .kpi-red::before{background:#EF4444;}
.kpi-purple::before{background:#8B5CF6;} .kpi-cyan::before{background:#06B6D4;}
.kpi-teal::before{background:#14B8A6;}
.kpi-label{font-size:11px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;}
.kpi-value{font-size:26px;font-weight:700;color:#F1F5F9;font-family:'DM Mono',monospace;line-height:1.1;}
.kpi-delta{font-size:12px;margin-top:4px;}
.kpi-delta.up{color:#10B981;} .kpi-delta.down{color:#EF4444;} .kpi-delta.flat{color:#64748B;}
.page-header{background:linear-gradient(135deg,#1E293B 0%,#0F172A 100%);border:1px solid #334155;border-radius:12px;padding:20px 24px;margin-bottom:20px;}
.page-title{font-size:22px;font-weight:700;color:#F1F5F9;margin:0;}
.page-sub{font-size:13px;color:#64748B;margin-top:3px;}
.section-hd{font-size:13px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:.07em;margin:18px 0 10px;border-bottom:1px solid #1E293B;padding-bottom:6px;}
.model-badge{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;margin-right:6px;}
.badge-xgb{background:#1D3A6E;color:#93C5FD;}
.badge-lstm{background:#3B1F6E;color:#C4B5FD;}
.badge-hybrid{background:#1A4A3A;color:#6EE7B7;}
.stDataFrame{border-radius:8px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

# ── COLOR PALETTE ──────────────────────────────────────────────────
PALETTE = ['#3B82F6','#10B981','#F59E0B','#EF4444',
           '#8B5CF6','#06B6D4','#EC4899','#F97316']

# ── FIX: xaxis/yaxis removed from PLOTLY_LAYOUT ───────────────────
# They are applied separately via apply_style() to prevent
# "multiple values for keyword argument" error
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#94A3B8', size=12),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8')),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor='#1E293B', font_color='#F1F5F9',
                    bordercolor='#334155'),
)
AXIS_STYLE = dict(gridcolor='#1E293B', linecolor='#334155',
                  tickfont=dict(color='#64748B'))

def apply_style(fig, height=340, **kwargs):
    """Apply standard dark theme to any Plotly figure. No xaxis/legend conflicts.
    Any key in kwargs overrides the same key in PLOTLY_LAYOUT."""
    layout = {**PLOTLY_LAYOUT, **kwargs, "height": height}
    fig.update_layout(**layout)
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig

# ── HELPER COMPONENTS ─────────────────────────────────────────────
def kpi(label, value, delta=None, color="blue"):
    dhtml = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        cls   = "up" if delta >= 0 else "down"
        dhtml = f'<div class="kpi-delta {cls}">{arrow} {abs(delta):.1f}% vs prev year</div>'
    st.markdown(f"""
    <div class="kpi-card kpi-{color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>{dhtml}
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-hd">{title}</div>', unsafe_allow_html=True)

def page_header(title, sub=""):
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">{title}</div>
      <div class="page-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def model_not_trained_banner():
    st.warning(
        "**Models not trained yet.**  \n"
        "Run `python train_all_models.py` from your project root folder first.  \n"
        "This trains XGBoost + LSTM + Hybrid and saves all results to `models/`."
    )

# ── DATA LOADING ───────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Order_Date"])
    df["Delivery_Date"]     = pd.to_datetime(df["Delivery_Date"],     errors="coerce")
    df["Expected_Delivery"] = pd.to_datetime(df["Expected_Delivery"], errors="coerce")
    df = df.sort_values("Order_Date").reset_index(drop=True)
    delivered = df["Delivery_Status"] == "Delivered"
    df["Delivery_Lead_Days"]  = np.where(delivered,
        (df["Delivery_Date"] - df["Order_Date"]).dt.days, np.nan)
    df["Delivery_Delay_Days"] = np.where(delivered,
        (df["Delivery_Date"] - df["Expected_Delivery"]).dt.days, np.nan)
    df["On_Time"] = np.where(delivered, df["Delivery_Delay_Days"] <= 0, np.nan)
    df["Week_Start"] = df["Order_Date"] - pd.to_timedelta(
        df["Order_Date"].dt.weekday, unit="D")
    df["Month_Num"] = df["Order_Date"].dt.month
    return df

@st.cache_resource
def load_xgb():
    p = model_path("xgb_model.pkl")
    if not os.path.exists(p): return None
    import joblib; return joblib.load(p)

@st.cache_resource
def load_lstm():
    p = model_path("lstm_model.h5")
    if not os.path.exists(p): return None
    from tensorflow.keras.models import load_model
    try:
        return load_model(p, compile=False)
    except Exception:
        return None

@st.cache_resource
def load_scaler_y():
    p = model_path("scaler_y.pkl")
    if not os.path.exists(p): return None
    import joblib; return joblib.load(p)

@st.cache_data
def load_metrics():
    p = model_path("model_metrics.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

@st.cache_data
def load_predictions():
    p = model_path("predictions_comparison.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p, parse_dates=["Week_Start"])
    return df

@st.cache_data
def load_feat_imp():
    p = model_path("feature_importance.csv")
    if not os.path.exists(p): return None
    return pd.read_csv(p)

@st.cache_data
def load_lstm_loss():
    p = model_path("lstm_loss_history.csv")
    if not os.path.exists(p): return None
    return pd.read_csv(p)

@st.cache_data
def load_weekly():
    p = model_path("weekly_series.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p, parse_dates=["Week_Start"])
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found. Update DATA_FILE in app.py.\nChecked: {DATA_PATH}")
    st.stop()

df_sales = df_raw[df_raw["Is_Return"] == "No"].copy()

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:16px;font-weight:700;color:#F1F5F9;margin-bottom:3px">📦 FMCG Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<span style="font-size:10px;background:#1D4ED8;color:#BFDBFE;padding:2px 8px;border-radius:20px;font-weight:500">Sales Forecasting System</span>', unsafe_allow_html=True)
    st.markdown("")

    page = st.radio("Navigation", [
        "🏠  Executive Dashboard",
        "📈  Sales Forecast",
        "🏢  Distributor Analysis",
        "📦  SKU & Category",
        "🚚  Delivery Intelligence",
        "🤖  Model Comparison",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Filters</div>', unsafe_allow_html=True)

    sel_dist = st.multiselect("Distributor",
        sorted(df_sales["Distributor_Name"].unique()),
        default=list(df_sales["Distributor_Name"].unique()))
    sel_year = st.multiselect("Year",
        sorted(df_sales["Year"].unique()),
        default=sorted(df_sales["Year"].unique()))
    sel_quarter = st.multiselect("Quarter",
        sorted(df_sales["Quarter"].unique()),
        default=sorted(df_sales["Quarter"].unique()))
    sel_zone = st.multiselect("Zone",
        sorted(df_sales["Zone"].unique()),
        default=list(df_sales["Zone"].unique()))
    sel_tier = st.multiselect("Tier",
        sorted(df_sales["Tier"].unique()),
        default=list(df_sales["Tier"].unique()))
    sel_beat = st.multiselect("Beat Type",
        sorted(df_sales["Beat_Key"].unique()),
        default=list(df_sales["Beat_Key"].unique()))
    sel_sbu = st.multiselect("SBU",
        sorted(df_sales["SBU"].unique()),
        default=list(df_sales["SBU"].unique()))
    sel_cat = st.multiselect("Category",
        sorted(df_sales["Category"].unique()),
        default=list(df_sales["Category"].unique()))

    st.markdown("---")
    st.caption(f"Dataset: {len(df_raw):,} rows · 2024–2025")

# ── FILTER ─────────────────────────────────────────────────────────
filt = df_sales[
    df_sales["Distributor_Name"].isin(sel_dist) &
    df_sales["Year"].isin(sel_year) &
    df_sales["Quarter"].isin(sel_quarter) &
    df_sales["Zone"].isin(sel_zone) &
    df_sales["Tier"].isin(sel_tier) &
    df_sales["Beat_Key"].isin(sel_beat) &
    df_sales["SBU"].isin(sel_sbu) &
    df_sales["Category"].isin(sel_cat)
].copy()

if filt.empty:
    st.warning("No data for current filters."); st.stop()

years     = sorted(filt["Year"].unique())
filt_cur  = filt[filt["Year"] == years[-1]] if len(years) > 1 else filt
filt_prev = filt[filt["Year"] == years[-2]] if len(years) > 1 else filt

def yoy(cur, prv, col, agg="sum"):
    c = cur[col].sum()  if agg=="sum" else cur[col].mean()
    p = prv[col].sum()  if agg=="sum" else prv[col].mean()
    return ((c-p)/p*100) if p!=0 else 0

MONTHS   = ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTHS_S = ["J","F","M","A","M","J","J","A","S","O","N","D"]

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "🏠  Executive Dashboard":

    page_header("Executive Dashboard",
                f"FMCG Sales Intelligence · {', '.join(str(y) for y in years)}")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: kpi("Total NSV",      f"₹{filt['Net_Sales_Value'].sum()/1e6:.2f}M",
                 yoy(filt_cur,filt_prev,"Net_Sales_Value"), "blue")
    with k2: kpi("Total Qty",      f"{filt['Quantity'].sum():,.0f}",
                 yoy(filt_cur,filt_prev,"Quantity"), "green")
    with k3: kpi("Avg Gross Margin",f"{filt['Gross_Margin_Pct'].mean():.1f}%", color="amber")
    with k4: kpi("Avg Achievement", f"{filt['Achievement_Pct'].mean():.1f}%", color="purple")
    with k5: kpi("Total Orders",    f"{len(filt):,}", color="cyan")
    with k6:
        ret_n = df_raw[df_raw["Distributor_Name"].isin(sel_dist) &
                       df_raw["Year"].isin(sel_year) &
                       (df_raw["Is_Return"]=="Yes")].__len__()
        kpi("Returns", f"{ret_n:,}", color="red")

    st.markdown("")
    c1, c2 = st.columns([2, 1])

    with c1:
        section("Monthly Net Sales Value Trend")
        monthly = filt.groupby(["Year","Month_Num"])["Net_Sales_Value"].sum().reset_index()
        fig = go.Figure()
        for i, yr in enumerate(sorted(monthly["Year"].unique())):
            m = monthly[monthly["Year"]==yr].sort_values("Month_Num")
            rgb = ','.join(str(int(PALETTE[i].lstrip('#')[j:j+2],16)) for j in (0,2,4))
            fig.add_trace(go.Scatter(
                x=m["Month_Num"], y=m["Net_Sales_Value"]/1e3,
                name=str(yr), mode="lines+markers",
                line=dict(color=PALETTE[i],width=2.5), marker=dict(size=6),
                fill="tozeroy", fillcolor=f"rgba({rgb},0.07)"))
        apply_style(fig, height=280, yaxis_title="NSV (₹K)")
        fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTHS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("SBU Revenue Share")
        sbu = filt.groupby("SBU")["Net_Sales_Value"].sum().reset_index()
        fig = go.Figure(go.Pie(labels=sbu["SBU"], values=sbu["Net_Sales_Value"],
            hole=0.55, marker_colors=PALETTE[:len(sbu)], textfont_size=11))
        apply_style(fig, height=280, showlegend=True,
                    legend=dict(orientation="v",x=1.0))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Beat Type Performance")
        beat_labels={"SWD":"SWD Route","GR2":"GR2 Route",
                     "ROC-A":"ROC-A","ROC-B":"ROC-B","2W":"2W Route","CMN":"Common Pace"}
        beat=(filt.groupby("Beat_Key").agg(NSV=("Net_Sales_Value","sum"))
              .reset_index().sort_values("NSV",ascending=True))
        beat["Label"] = beat["Beat_Key"].map(beat_labels).fillna(beat["Beat_Key"])
        fig = go.Figure(go.Bar(y=beat["Label"], x=beat["NSV"]/1e3,
            orientation="h", marker_color=PALETTE[0],
            hovertemplate="<b>%{y}</b><br>₹%{x:,.1f}K<extra></extra>"))
        apply_style(fig, height=280, xaxis_title="NSV (₹K)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Zone × Tier Revenue Heatmap")
        pivot = filt.pivot_table(values="Net_Sales_Value",index="Zone",
                                  columns="Tier",aggfunc="sum",fill_value=0)/1e3
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
            colorscale=[[0,"#0F172A"],[0.5,"#1D4ED8"],[1,"#3B82F6"]],
            text=[[f"₹{v:.0f}K" for v in row] for row in pivot.values],
            texttemplate="%{text}", textfont_size=11))
        apply_style(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section("Category Revenue Breakdown")
        cat = filt.groupby("Category")["Net_Sales_Value"].sum().reset_index().sort_values("Net_Sales_Value",ascending=False)
        fig = px.bar(cat, x="Category", y="Net_Sales_Value",
                     color="Net_Sales_Value",
                     color_continuous_scale=["#0F172A","#3B82F6"])
        apply_style(fig, height=280, coloraxis_showscale=False)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Quarterly NSV Comparison")
        qtr = filt.groupby(["Year","Quarter"])["Net_Sales_Value"].sum().reset_index()
        fig = px.bar(qtr, x="Quarter", y="Net_Sales_Value", color="Year",
                     barmode="group", color_discrete_sequence=PALETTE[:2])
        apply_style(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON  ★ NEW ★
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  Model Comparison":

    page_header("Model Comparison",
                "XGBoost vs LSTM vs Hybrid · RMSE · MAE · R² · Feature Importance")

    metrics  = load_metrics()
    preds_df = load_predictions()
    feat_imp = load_feat_imp()
    loss_df  = load_lstm_loss()

    if metrics is None:
        model_not_trained_banner()
        st.markdown("""
**How to train all three models in one step:**
```bash
# From your project root folder:
cd "E:\\Imarticus BNGLR\\FMCG Sales Forecsting 24-25"
python train_all_models.py
```
This takes about 3–5 minutes and saves everything to `models/`.
        """)
        st.stop()

    # Model badges
    st.markdown("""
    <span class="model-badge badge-xgb">XGBoost · Machine Learning</span>
    <span class="model-badge badge-lstm">LSTM · Deep Learning</span>
    <span class="model-badge badge-hybrid">Hybrid · Combined (60% XGB + 40% LSTM)</span>
    """, unsafe_allow_html=True)
    st.markdown("")

    # KPI row — best metric highlighted
    models_list = ["XGBoost","LSTM","Hybrid"]
    rmse_vals = [metrics[m]["RMSE"] for m in models_list]
    mae_vals  = [metrics[m]["MAE"]  for m in models_list]
    r2_vals   = [metrics[m]["R2"]   for m in models_list]

    best_rmse = models_list[int(np.argmin(rmse_vals))]
    best_r2   = models_list[int(np.argmax(r2_vals))]

    k1,k2,k3 = st.columns(3)
    for col, mname, rmse, mae, r2 in zip(
        [k1,k2,k3], models_list, rmse_vals, mae_vals, r2_vals
    ):
        color = "green" if mname == best_rmse else "blue" if mname=="XGBoost" else "purple"
        with col:
            kpi(f"{mname} · RMSE",
                f"₹{rmse:,.0f}",
                color=color)
    st.markdown("")

    k1,k2,k3 = st.columns(3)
    for col, mname, mae in zip([k1,k2,k3], models_list, mae_vals):
        color = "green" if mname == best_rmse else "blue" if mname=="XGBoost" else "purple"
        with col:
            kpi(f"{mname} · MAE", f"₹{mae:,.0f}", color=color)
    st.markdown("")

    k1,k2,k3 = st.columns(3)
    for col, mname, r2 in zip([k1,k2,k3], models_list, r2_vals):
        color = "green" if mname == best_r2 else "blue" if mname=="XGBoost" else "purple"
        with col:
            kpi(f"{mname} · R²", f"{r2:.4f}", color=color)
    st.markdown("")

    # Metrics comparison bar charts
    section("Metrics Comparison — All Three Models")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = go.Figure(go.Bar(
            x=models_list, y=rmse_vals,
            marker_color=[PALETTE[1] if m==best_rmse else PALETTE[0]
                          for m in models_list],
            text=[f"₹{v:,.0f}" for v in rmse_vals],
            textposition="outside"))
        apply_style(fig, height=260, yaxis_title="RMSE (₹)")
        fig.update_layout(title_text="RMSE  ↓ lower is better")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=models_list, y=mae_vals,
            marker_color=[PALETTE[1] if m==best_rmse else PALETTE[2]
                          for m in models_list],
            text=[f"₹{v:,.0f}" for v in mae_vals],
            textposition="outside"))
        apply_style(fig, height=260, yaxis_title="MAE (₹)")
        fig.update_layout(title_text="MAE  ↓ lower is better")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = go.Figure(go.Bar(
            x=models_list, y=r2_vals,
            marker_color=[PALETTE[1] if m==best_r2 else PALETTE[4]
                          for m in models_list],
            text=[f"{v:.4f}" for v in r2_vals],
            textposition="outside"))
        apply_style(fig, height=260, yaxis_title="R² Score",
                    yaxis_range=[max(0, min(r2_vals)-0.05), 1.0])
        fig.update_layout(title_text="R²  ↑ higher is better")
        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted chart
    if preds_df is not None:
        section("Actual vs Predicted — Weekly NSV (Test Set)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preds_df["Week_Start"], y=preds_df["Actual_NSV"]/1e3,
            mode="lines", name="Actual",
            line=dict(color="#F1F5F9", width=2.5)))
        fig.add_trace(go.Scatter(
            x=preds_df["Week_Start"], y=preds_df["XGB_Pred"]/1e3,
            mode="lines", name="XGBoost",
            line=dict(color=PALETTE[0], width=2, dash="dash")))
        if "LSTM_Pred" in preds_df.columns:
            fig.add_trace(go.Scatter(
                x=preds_df["Week_Start"], y=preds_df["LSTM_Pred"]/1e3,
                mode="lines", name="LSTM",
                line=dict(color=PALETTE[3], width=2, dash="dot")))
        if "Hybrid_Pred" in preds_df.columns:
            fig.add_trace(go.Scatter(
                x=preds_df["Week_Start"], y=preds_df["Hybrid_Pred"]/1e3,
                mode="lines", name="Hybrid",
                line=dict(color=PALETTE[1], width=2.5, dash="dashdot")))
        apply_style(fig, height=360, yaxis_title="Weekly NSV (₹K)")
        fig.update_layout(legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance + LSTM loss
    c1, c2 = st.columns(2)

    with c1:
        if feat_imp is not None:
            section("XGBoost — Top 15 Feature Importances")
            top15 = feat_imp.head(15)
            fig = go.Figure(go.Bar(
                y=top15["Feature"][::-1], x=top15["Importance"][::-1],
                orientation="h", marker_color=PALETTE[0],
                hovertemplate="<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>"))
            apply_style(fig, height=400, xaxis_title="Importance Score")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if loss_df is not None:
            section("LSTM — Training vs Validation Loss")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=loss_df["Epoch"], y=loss_df["Train_Loss"],
                mode="lines", name="Train Loss",
                line=dict(color=PALETTE[0], width=2)))
            fig.add_trace(go.Scatter(
                x=loss_df["Epoch"], y=loss_df["Val_Loss"],
                mode="lines", name="Val Loss",
                line=dict(color=PALETTE[3], width=2, dash="dash")))
            apply_style(fig, height=400,
                        xaxis_title="Epoch", yaxis_title="MSE Loss")
            fig.update_layout(legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)

    # Training summary table
    section("Training Summary")
    info = metrics.get("training_info", {})
    if info:
        summary = pd.DataFrame([
            ("Dataset rows",      f"{info.get('dataset_rows','-'):,}"),
            ("Train rows (80%)",  f"{info.get('train_rows','-'):,}"),
            ("Test rows (20%)",   f"{info.get('test_rows','-'):,}"),
            ("ML features",       str(info.get("features_count","-"))),
            ("Weekly data points",str(info.get("weekly_points","-"))),
            ("LSTM window size",  f"{info.get('lstm_window',8)} weeks"),
            ("LSTM train seqs",   str(info.get("lstm_train_seqs","-"))),
            ("LSTM test seqs",    str(info.get("lstm_test_seqs","-"))),
        ], columns=["Parameter","Value"])
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — SALES FORECAST
# ══════════════════════════════════════════════════════════════════
elif page == "📈  Sales Forecast":

    page_header("Sales Forecast Engine",
                "Trend analysis · Seasonal patterns · ML-powered projections")

    metrics  = load_metrics()
    weekly   = load_weekly()
    preds_df = load_predictions()

    fc1, fc2, fc3 = st.columns([1,1,2])
    with fc1:
        fc_horizon = st.selectbox("Forecast Horizon",
                                   [2,4,6,8,10,12], index=2,
                                   format_func=lambda x: f"{x} months")
    with fc2:
        fc_level = st.selectbox("Breakdown by",
                                 ["Overall","SBU","Category","Beat Type","SKU"])
    with fc3:
        st.caption(
            "💡 Checkpoints every 2 months act as review points — "
            "use them to plan SKU-level promotions or distributor "
            "schemes if a checkpoint shows de-growth.")

    # Monthly series from filtered data
    monthly = (filt.groupby(["Year","Month_Num"])["Net_Sales_Value"]
               .sum().reset_index().sort_values(["Year","Month_Num"]))
    monthly["Period"] = pd.to_datetime(
        monthly[["Year","Month_Num"]].rename(
            columns={"Year":"year","Month_Num":"month"}).assign(day=1))
    monthly = monthly.sort_values("Period").reset_index(drop=True)

    # ── Forecast logic ─────────────────────────────────────────────
    def make_forecast(series_vals, horizon, ref_monthly):
        if len(series_vals) < 3:
            return pd.DataFrame({"Period":[], "Forecast":[]})
        recent = series_vals[-7:] if len(series_vals) >= 7 else series_vals
        grates = np.diff(recent) / (np.abs(recent[:-1]) + 1e-9)
        avg_g  = np.clip(np.mean(grates), -0.15, 0.20)
        m_avg  = (ref_monthly.groupby("Month_Num")["Net_Sales_Value"]
                  .mean().reindex(range(1,13)).ffill())
        overall = m_avg.mean()
        seas = (m_avg / overall).fillna(1)
        last_val  = series_vals[-1]
        last_date = monthly["Period"].iloc[-1]
        preds, dates = [], []
        val = last_val
        for i in range(1, horizon+1):
            nd = last_date + pd.DateOffset(months=i)
            val = val * (1 + avg_g) * seas.get(nd.month, 1.0)
            preds.append(val)
            dates.append(nd)
        return pd.DataFrame({"Period": dates, "Forecast": preds})

    def suggest_action(growth):
        if growth >= 5:
            return "✅ On track — maintain plan"
        elif growth >= 0:
            return "🟡 Flat — consider light promo"
        elif growth >= -8:
            return "⚠️ De-growth — plan scheme"
        else:
            return "🔴 Sharp de-growth — review mix"

    fc_df = make_forecast(monthly["Net_Sales_Value"].values, fc_horizon, monthly)
    if not fc_df.empty:
        std = monthly["Net_Sales_Value"].pct_change().dropna().std()
        fc_df["Upper"] = fc_df["Forecast"] * (1 + std)
        fc_df["Lower"] = fc_df["Forecast"] * (1 - std)

    # ── XGBoost-calibrated forecast ─────────────────────────────────
    # XGBoost showed the strongest accuracy in Model Comparison, so we
    # use its demonstrated prediction-to-actual ratio (from the held-out
    # test weeks) to calibrate the trend+seasonal forecast above.
    xgb_fc_df = None
    if preds_df is not None and not fc_df.empty and \
       preds_df["Actual_NSV"].sum() != 0:
        xgb_ratio = preds_df["XGB_Pred"].sum() / preds_df["Actual_NSV"].sum()
        xgb_ratio = float(np.clip(xgb_ratio, 0.85, 1.15))  # sanity bounds
        xgb_fc_df = fc_df[["Period"]].copy()
        xgb_fc_df["XGB_Forecast"] = fc_df["Forecast"] * xgb_ratio

    # ── Main forecast chart ────────────────────────────────────────
    section("Historical NSV + Forecast Projection")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Period"], y=monthly["Net_Sales_Value"]/1e3,
        mode="lines+markers", name="Actual NSV",
        line=dict(color=PALETTE[0],width=2.5), marker=dict(size=5)))
    roll = monthly["Net_Sales_Value"].rolling(3).mean()
    fig.add_trace(go.Scatter(
        x=monthly["Period"], y=roll/1e3,
        mode="lines", name="3-Month Rolling Avg",
        line=dict(color=PALETTE[2],width=1.5,dash="dot")))
    if not fc_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["Period"],fc_df["Period"][::-1]]),
            y=pd.concat([fc_df["Upper"]/1e3,fc_df["Lower"][::-1]/1e3]),
            fill="toself", fillcolor="rgba(16,185,129,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=True, name="Confidence Band"))
        fig.add_trace(go.Scatter(
            x=fc_df["Period"], y=fc_df["Forecast"]/1e3,
            mode="lines+markers", name="Trend Forecast",
            line=dict(color=PALETTE[1],width=2.5,dash="dash"),
            marker=dict(size=8,symbol="diamond")))
    if xgb_fc_df is not None:
        fig.add_trace(go.Scatter(
            x=xgb_fc_df["Period"], y=xgb_fc_df["XGB_Forecast"]/1e3,
            mode="lines+markers", name="XGBoost-Calibrated Forecast",
            line=dict(color=PALETTE[3],width=2.5,dash="dot"),
            marker=dict(size=7,symbol="square")))
    _split_x = monthly["Period"].max()
    fig.add_shape(type="line", x0=_split_x, x1=_split_x,
                  y0=0, y1=1, yref="paper",
                  line=dict(color="#475569", dash="dot", width=1.5))
    fig.add_annotation(x=_split_x, y=1, yref="paper",
                        text="Forecast →", showarrow=False,
                        font=dict(color="#94A3B8", size=11),
                        xanchor="left", yanchor="bottom")
    apply_style(fig, height=380, yaxis_title="NSV (₹K)")
    fig.update_layout(legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast KPI strip ────────────────────────────────────────
    f1,f2,f3,f4 = st.columns(4)
    if not fc_df.empty and len(monthly) >= 3:
        baseline = monthly['Net_Sales_Value'].tail(3).mean()
        total_fc = fc_df['Forecast'].sum()
        avg_fc   = fc_df['Forecast'].mean()
        g = (avg_fc - baseline) / (baseline + 1e-9) * 100
        peak_row = fc_df.loc[fc_df['Forecast'].idxmax()]
        peak_label = peak_row['Period'].strftime('%b %Y')
    else:
        total_fc = avg_fc = g = 0.0
        peak_label = "—"

    with f1: kpi(f"Total Forecast ({fc_horizon}M)", f"₹{total_fc/1e6:.2f}M", color="blue")
    with f2: kpi("Avg Monthly Forecast", f"₹{avg_fc/1e3:.1f}K", color="green")
    with f3: kpi("Expected Growth", f"{g:+.1f}%",
                 color="green" if g>=0 else "red")
    with f4: kpi("Peak Forecast Month", peak_label, color="purple")

    st.markdown("")

    # ── Forecast Action Plan — bi-monthly checkpoints ──────────────
    if not fc_df.empty:
        section("Forecast Action Plan — Review Checkpoints")
        st.caption(
            "Each checkpoint compares the cumulative forecast for that "
            "period against the recent monthly run-rate, so de-growth "
            "can be flagged early — react with SKU-level promos, "
            "distributor schemes, or beat-coverage reviews before it compounds."
        )
        baseline_monthly = monthly["Net_Sales_Value"].tail(3).mean()
        checkpoints = [c for c in [2,4,6,8,10,12] if c <= fc_horizon]
        action_rows = []
        for cp in checkpoints:
            window = fc_df.iloc[:cp]
            cum_fc = window["Forecast"].sum()
            avg_fc = window["Forecast"].mean()
            growth = (avg_fc - baseline_monthly) / (baseline_monthly + 1e-9) * 100
            if growth >= 5:
                action = "✅ On track — maintain current plan"
            elif growth >= 0:
                action = "🟡 Flat growth — consider light SKU promo"
            elif growth >= -8:
                action = "⚠️ Mild de-growth — plan distributor scheme"
            else:
                action = "🔴 Significant de-growth — review beat coverage & SKU mix"
            tag = " (Quarter-end)" if cp % 3 == 0 else ""
            action_rows.append({
                "Checkpoint": f"{cp} months{tag}",
                "Cumulative Forecast NSV": f"₹{cum_fc:,.0f}",
                "Avg Monthly Forecast": f"₹{avg_fc:,.0f}",
                "Growth vs Recent Avg": f"{growth:+.1f}%",
                "Suggested Action": action,
            })
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    st.markdown("")

    # ── Breakdown forecast by dimension ───────────────────────────
    if fc_level != "Overall":
        section(f"Forecast by {fc_level}")
        dim_col = {"SBU":"SBU","Category":"Category",
                   "Beat Type":"Beat_Key","SKU":"SKU_Name"}[fc_level]
        dim_monthly = (filt.groupby([dim_col,"Year","Month_Num"])["Net_Sales_Value"]
                       .sum().reset_index())
        dim_values = filt[dim_col].unique()
        if fc_level == "SKU":
            # limit to top 15 SKUs by total NSV to keep the table readable
            dim_values = (filt.groupby(dim_col)["Net_Sales_Value"].sum()
                           .sort_values(ascending=False).head(15).index)
            st.caption("Showing top 15 SKUs by total NSV.")
        rows = []
        for dv in dim_values:
            sub = dim_monthly[dim_monthly[dim_col]==dv].sort_values(["Year","Month_Num"])
            if len(sub)<3: continue
            fc_s = make_forecast(sub["Net_Sales_Value"].values, fc_horizon, monthly)
            if fc_s.empty: continue
            g = (fc_s["Forecast"].mean()-sub["Net_Sales_Value"].tail(3).mean()) / \
                (sub["Net_Sales_Value"].tail(3).mean()+1e-9)*100
            rows.append({dim_col: dv,
                "Last 3M Avg": f"₹{sub['Net_Sales_Value'].tail(3).mean():,.0f}",
                f"Forecast Avg ({fc_horizon}M)": f"₹{fc_s['Forecast'].mean():,.0f}",
                "Total Forecast": f"₹{fc_s['Forecast'].sum():,.0f}",
                "Growth %": f"{g:+.1f}%",
                "Suggested Action": suggest_action(g)})
        if rows:
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    # ── YoY comparison ─────────────────────────────────────────────
    section("Year-over-Year Monthly Comparison + Forecast")
    if len(years) > 1:
        yoy_df = filt.groupby(["Year","Month_Num"])["Net_Sales_Value"].sum().unstack(0).fillna(0)
        fig = go.Figure()
        for i, yr in enumerate(yoy_df.columns):
            fig.add_trace(go.Scatter(
                x=list(range(1,13)), y=yoy_df[yr].values/1e3,
                mode="lines+markers", name=str(yr),
                line=dict(color=PALETTE[i],width=2.5), marker=dict(size=6)))

        # Extend the latest year with the trend forecast for remaining months
        if not fc_df.empty:
            last_year = years[-1]
            last_month_done = int(monthly[monthly["Year"]==last_year]["Month_Num"].max())
            fc_for_year = fc_df[fc_df["Period"].dt.year == last_year].copy()
            if not fc_for_year.empty:
                fc_for_year["Month_Num"] = fc_for_year["Period"].dt.month
                # connect from last actual point
                x_conn = [last_month_done] + fc_for_year["Month_Num"].tolist()
                y_conn = [yoy_df[last_year].loc[last_month_done]/1e3] + \
                         (fc_for_year["Forecast"]/1e3).tolist()
                fig.add_trace(go.Scatter(
                    x=x_conn, y=y_conn,
                    mode="lines+markers", name=f"{last_year} Forecast",
                    line=dict(color=PALETTE[len(yoy_df.columns) % len(PALETTE)],
                               width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond")))
            # also project rollover months into next year if forecast spans across
            next_year = last_year + 1
            fc_for_next = fc_df[fc_df["Period"].dt.year == next_year].copy()
            if not fc_for_next.empty:
                fc_for_next["Month_Num"] = fc_for_next["Period"].dt.month
                fig.add_trace(go.Scatter(
                    x=fc_for_next["Month_Num"], y=fc_for_next["Forecast"]/1e3,
                    mode="lines+markers", name=f"{next_year} Forecast",
                    line=dict(color=PALETTE[(len(yoy_df.columns)+1) % len(PALETTE)],
                               width=2.5, dash="dash"),
                    marker=dict(size=7, symbol="diamond")))

        apply_style(fig, height=300, yaxis_title="NSV (₹K)")
        fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTHS)
        fig.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Dashed lines extend the trend+seasonal forecast for the "
            "current year (and into next year where the horizon spans "
            "year-end), so YoY momentum is visible alongside the projection."
        )

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — DISTRIBUTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "🏢  Distributor Analysis":

    page_header("Distributor Performance Analysis",
                "Revenue ranking · Achievement tracking · Beat intelligence")

    dist_agg = filt.groupby("Distributor_Name").agg(
        NSV=("Net_Sales_Value","sum"), Qty=("Quantity","sum"),
        Orders=("Order_ID","count"), Avg_Margin=("Gross_Margin_Pct","mean"),
        Avg_Achieve=("Achievement_Pct","mean"),
        Avg_Discount=("Discount_Pct","mean")
    ).reset_index().sort_values("NSV",ascending=False)

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Active Distributors", f"{len(dist_agg)}", color="blue")
    with k2: kpi("Total NSV", f"₹{dist_agg['NSV'].sum()/1e6:.2f}M", color="green")
    with k3: kpi("Avg Achievement",f"{dist_agg['Avg_Achieve'].mean():.1f}%",color="amber")
    with k4: kpi("Avg Gross Margin",f"{dist_agg['Avg_Margin'].mean():.1f}%",color="purple")
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        section("Distributor Revenue Ranking")
        top = dist_agg.sort_values("NSV",ascending=True).tail(15)
        fig = go.Figure(go.Bar(
            y=top["Distributor_Name"], x=top["NSV"]/1e3,
            orientation="h", marker_color=PALETTE[0],
            text=[f"₹{v/1e3:.1f}K" for v in top["NSV"]],
            textposition="outside"))
        apply_style(fig, height=400, xaxis_title="NSV (₹K)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("Achievement % vs Target (92% line)")
        sa = dist_agg.sort_values("Avg_Achieve",ascending=True)
        clrs=[PALETTE[1] if v>=92 else PALETTE[2] if v>=85 else PALETTE[3]
              for v in sa["Avg_Achieve"]]
        fig = go.Figure(go.Bar(
            y=sa["Distributor_Name"], x=sa["Avg_Achieve"],
            orientation="h", marker_color=clrs,
            text=[f"{v:.1f}%" for v in sa["Avg_Achieve"]],
            textposition="outside"))
        fig.add_vline(x=92,line_dash="dot",line_color="#F59E0B",
                      annotation_text="92% target",
                      annotation_font_color="#F59E0B")
        apply_style(fig, height=400, xaxis_title="Achievement (%)")
        st.plotly_chart(fig, use_container_width=True)

    section("Beat Type NSV by Distributor")
    beat_dist=(filt.groupby(["Distributor_Name","Beat_Key"])["Net_Sales_Value"]
               .sum().unstack("Beat_Key").fillna(0)/1e3)
    fig = go.Figure(go.Heatmap(
        z=beat_dist.values, x=list(beat_dist.columns), y=list(beat_dist.index),
        colorscale=[[0,"#0F172A"],[0.4,"#1E40AF"],[1,"#3B82F6"]],
        text=[[f"₹{v:.0f}K" for v in row] for row in beat_dist.values],
        texttemplate="%{text}", textfont_size=10))
    apply_style(fig, height=340)
    st.plotly_chart(fig, use_container_width=True)

    section("Monthly NSV — Top 5 Distributors")
    top5 = dist_agg.head(5)["Distributor_Name"].tolist()
    t5m  = (filt[filt["Distributor_Name"].isin(top5)]
            .groupby(["Distributor_Name","Year","Month_Num"])["Net_Sales_Value"]
            .sum().reset_index())
    fig = px.line(t5m, x="Month_Num", y="Net_Sales_Value",
                  color="Distributor_Name", facet_col="Year",
                  color_discrete_sequence=PALETTE, markers=True)
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTHS_S)
    apply_style(fig, height=320)
    st.plotly_chart(fig, use_container_width=True)

    section("Distributor Metrics Table")
    disp = dist_agg.copy()
    disp["NSV"]         = disp["NSV"].map(lambda x: f"₹{x:,.0f}")
    disp["Qty"]         = disp["Qty"].map(lambda x: f"{x:,.0f}")
    disp["Avg_Margin"]  = disp["Avg_Margin"].map(lambda x: f"{x:.1f}%")
    disp["Avg_Achieve"] = disp["Avg_Achieve"].map(lambda x: f"{x:.1f}%")
    disp["Avg_Discount"]= disp["Avg_Discount"].map(lambda x: f"{x:.1f}%")
    disp.columns = ["Distributor","NSV","Qty","Orders",
                    "Avg Margin","Avg Achievement","Avg Discount"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 — SKU & CATEGORY
# ══════════════════════════════════════════════════════════════════
elif page == "📦  SKU & Category":

    page_header("SKU & Category Intelligence",
                "Product performance · Margin analysis · Slow-movers")

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Active SKUs",  f"{filt['SKU_Code'].nunique()}", color="blue")
    with k2: kpi("Categories",   f"{filt['Category'].nunique()}",  color="green")
    with k3: kpi("Avg MRP",      f"₹{filt['MRP'].mean():.0f}",    color="amber")
    with k4: kpi("Avg Discount", f"{filt['Discount_Pct'].mean():.1f}%", color="purple")
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        section("Top 15 SKUs by Revenue")
        sku_rev=(filt.groupby("SKU_Name")["Net_Sales_Value"]
                 .sum().sort_values(ascending=True).tail(15).reset_index())
        fig = go.Figure(go.Bar(y=sku_rev["SKU_Name"],x=sku_rev["Net_Sales_Value"]/1e3,
            orientation="h",marker_color=PALETTE[0]))
        apply_style(fig,height=400,xaxis_title="NSV (₹K)")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        section("Category Revenue & Margin")
        cat_agg=filt.groupby("Category").agg(
            NSV=("Net_Sales_Value","sum"),
            Margin=("Gross_Margin_Pct","mean")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=cat_agg["Category"],y=cat_agg["NSV"]/1e3,
            name="NSV (₹K)",marker_color=PALETTE[0]),secondary_y=False)
        fig.add_trace(go.Scatter(x=cat_agg["Category"],y=cat_agg["Margin"],
            mode="lines+markers",name="Avg Margin %",
            line=dict(color=PALETTE[1],width=2.5),marker=dict(size=7)),
            secondary_y=True)
        apply_style(fig,height=400)
        fig.update_xaxes(tickangle=-35)
        fig.update_layout(legend=dict(orientation="h",y=1.08))
        fig.update_yaxes(title_text="NSV (₹K)",secondary_y=False,gridcolor="#1E293B")
        fig.update_yaxes(title_text="Margin (%)",secondary_y=True)
        st.plotly_chart(fig,use_container_width=True)

    section("SBU Monthly Revenue Trend")
    sbu_m=(filt.groupby(["SBU","Year","Month_Num"])["Net_Sales_Value"].sum().reset_index())
    sbu_m["Period"]=sbu_m.apply(
        lambda r:pd.Timestamp(year=int(r["Year"]),month=int(r["Month_Num"]),day=1),axis=1)
    fig=px.area(sbu_m.sort_values("Period"),x="Period",y="Net_Sales_Value",
                color="SBU",color_discrete_sequence=PALETTE[:3])
    apply_style(fig,height=280)
    st.plotly_chart(fig,use_container_width=True)

    section("Discount % vs Gross Margin % (per SKU)")
    ss=filt.groupby("SKU_Name").agg(
        Discount=("Discount_Pct","mean"),
        Margin=("Gross_Margin_Pct","mean"),
        NSV=("Net_Sales_Value","sum")).reset_index()
    fig=px.scatter(ss,x="Discount",y="Margin",size="NSV",
        color="Margin",text="SKU_Name",
        color_continuous_scale=["#EF4444","#F59E0B","#10B981"],
        size_max=30)
    fig.update_traces(textposition="top center",textfont_size=8)
    apply_style(fig,height=380,coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        section("Top Moving SKUs (by Quantity)")
        top_mov=(filt.groupby("SKU_Name")["Quantity"]
              .sum().sort_values(ascending=False).head(10).reset_index())
        top_mov.columns=["SKU","Total Qty"]; top_mov["Status"]="🚀 Top Mover"
        st.dataframe(top_mov,use_container_width=True,hide_index=True)

    with c2:
        section("Slow-Moving SKUs (Bottom 10)")
        slow=(filt.groupby("SKU_Name")["Quantity"]
              .sum().sort_values().head(10).reset_index())
        slow.columns=["SKU","Total Qty"]; slow["Status"]="⚠️ Slow"
        st.dataframe(slow,use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 6 — DELIVERY INTELLIGENCE
# ══════════════════════════════════════════════════════════════════
elif page == "🚚  Delivery Intelligence":

    page_header("Delivery Intelligence",
                "Lead time · Delay analysis · Payment status · Beat reliability")

    delivered = filt[filt["Delivery_Status"]=="Delivered"].copy()
    total = len(filt)
    dlv_n = (filt["Delivery_Status"]=="Delivered").sum()
    can_n = (filt["Delivery_Status"]=="Cancelled").sum()
    on_n  = delivered["On_Time"].sum() if "On_Time" in delivered else 0
    avgl  = delivered["Delivery_Lead_Days"].mean() if "Delivery_Lead_Days" in delivered.columns else 0
    avgd  = delivered["Delivery_Delay_Days"].mean() if "Delivery_Delay_Days" in delivered.columns else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Delivery Rate",     f"{dlv_n/total*100:.1f}%", color="green")
    with k2: kpi("On-Time Rate",      f"{on_n/max(dlv_n,1)*100:.1f}%", color="blue")
    with k3: kpi("Avg Lead Time",     f"{avgl:.1f} days", color="amber")
    with k4: kpi("Avg Delay",         f"{avgd:+.1f} days",
                 color="red" if avgd>0 else "green")
    with k5: kpi("Cancellation Rate", f"{can_n/total*100:.1f}%", color="purple")
    st.markdown("")

    c1,c2=st.columns(2)
    with c1:
        section("Delivery Status Distribution")
        ds=filt["Delivery_Status"].value_counts().reset_index()
        ds.columns=["Status","Count"]
        cmap={"Delivered":PALETTE[1],"Cancelled":PALETTE[3],
              "In Transit":PALETTE[2],"Pending":PALETTE[4]}
        fig=go.Figure(go.Pie(labels=ds["Status"],values=ds["Count"],hole=0.5,
            marker_colors=[cmap.get(s,PALETTE[0]) for s in ds["Status"]],
            textfont_size=11))
        apply_style(fig,height=280)
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        section("Payment Status Distribution")
        ps=filt["Payment_Status"].value_counts().reset_index()
        ps.columns=["Status","Count"]
        pmap={"Paid":PALETTE[1],"Pending":PALETTE[2],
              "Overdue":PALETTE[3],"Refunded":PALETTE[4]}
        fig=go.Figure(go.Bar(x=ps["Status"],y=ps["Count"],
            marker_color=[pmap.get(s,PALETTE[0]) for s in ps["Status"]],
            text=ps["Count"],textposition="outside"))
        apply_style(fig,height=280)
        st.plotly_chart(fig,use_container_width=True)

    if not delivered.empty and "Delivery_Lead_Days" in delivered.columns:
        section("Delivery Lead Time Distribution (Delivered Orders)")
        lc=delivered["Delivery_Lead_Days"].dropna()
        fig=go.Figure()
        fig.add_trace(go.Histogram(x=lc,nbinsx=20,
            marker_color=PALETTE[0],marker_line_color="#0F172A",
            marker_line_width=1))
        fig.add_vline(x=lc.mean(),line_dash="dot",line_color=PALETTE[2],
                      annotation_text=f"Avg:{lc.mean():.1f}d",
                      annotation_font_color=PALETTE[2])
        apply_style(fig,height=260,xaxis_title="Lead Time (Days)",
                    yaxis_title="Order Count")
        st.plotly_chart(fig,use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        section("Delivery Rate by Beat Type")
        bd=(filt.groupby("Beat_Key")
            .apply(lambda x:(x["Delivery_Status"]=="Delivered").mean()*100)
            .reset_index(name="Delivery_Rate")
            .sort_values("Delivery_Rate",ascending=True))
        fig=go.Figure(go.Bar(y=bd["Beat_Key"],x=bd["Delivery_Rate"],
            orientation="h",
            marker_color=[PALETTE[1] if v>=60 else PALETTE[2] if v>=45
                          else PALETTE[3] for v in bd["Delivery_Rate"]],
            text=[f"{v:.1f}%" for v in bd["Delivery_Rate"]],
            textposition="outside"))
        apply_style(fig,height=280,xaxis_title="Delivery Rate (%)")
        st.plotly_chart(fig,use_container_width=True)

    with c2:
        if not delivered.empty and "On_Time" in delivered.columns:
            section("Monthly On-Time Delivery Trend")
            ot=(delivered.groupby(["Year","Month_Num"])["On_Time"]
                .mean()*100).reset_index()
            ot["Period"]=ot.apply(
                lambda r:pd.Timestamp(year=int(r["Year"]),month=int(r["Month_Num"]),day=1),axis=1)
            fig=go.Figure()
            for i,yr in enumerate(sorted(ot["Year"].unique())):
                m=ot[ot["Year"]==yr].sort_values("Month_Num")
                fig.add_trace(go.Scatter(x=m["Month_Num"],y=m["On_Time"],
                    mode="lines+markers",name=str(yr),
                    line=dict(color=PALETTE[i],width=2.5),marker=dict(size=6)))
            fig.add_hline(y=80,line_dash="dot",line_color="#F59E0B",
                          annotation_text="80% target",
                          annotation_font_color="#F59E0B")
            apply_style(fig,height=280,yaxis_title="On-Time Rate (%)")
            fig.update_xaxes(tickvals=list(range(1,13)),ticktext=MONTHS_S)
            st.plotly_chart(fig,use_container_width=True)

    section("Outstanding Payments by Distributor")
    ov=filt[filt["Payment_Status"].isin(["Overdue","Pending"])]
    if not ov.empty:
        ovd=(ov.groupby(["Distributor_Name","Payment_Status"])
             ["Net_Sales_Value"].sum().reset_index())
        fig=px.bar(ovd,x="Distributor_Name",y="Net_Sales_Value",
                   color="Payment_Status",
                   color_discrete_map={"Overdue":PALETTE[3],"Pending":PALETTE[2]},
                   barmode="stack")
        apply_style(fig,height=300)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.success("No outstanding payments for selected filters.")
