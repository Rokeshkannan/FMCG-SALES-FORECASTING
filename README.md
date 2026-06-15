# 📦 FMCG Sales Forecasting System
### Hybrid AI-Powered Sales Intelligence Dashboard — XGBoost + LSTM + Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fmcg-sales-forecasting-system2.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-red)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔍 Project Overview

This project builds a **production-ready FMCG Sales Forecasting System** for a Tamil Nadu-based FMCG distribution network. It combines Machine Learning (XGBoost) and Deep Learning (LSTM) into a **Hybrid forecasting model**, all wrapped in an interactive Streamlit dashboard for real business decision-making.

The system models real-world FMCG distribution logic — **SWD, GR2, ROC, 2W, and Common Pace beats** — with each distributor having one DS (Delivery Salesman) covering 6 beats, each serving strictly defined outlet types.

> **Live Demo →** [fmcg-sales-forecasting-system2.streamlit.app](https://fmcg-sales-forecasting-system2.streamlit.app/)

---

## 🎯 Problem Statement

FMCG sales forecasting is challenging because performance is influenced by:
- Beat-level route structure and outlet types
- Seasonal demand fluctuations
- SKU-level discount and margin variability
- Distributor tier performance
- Sequential demand patterns over time

Traditional reporting tools show **what happened** — this system predicts **what will happen** and suggests **what to do about it**.

---

## 🏗️ System Architecture

```
Raw Sales Data (15,503 rows · 47 columns · 2024–2025)
         │
         ▼
  Feature Engineering
  (Lag features · Rolling averages · Cyclic encoding)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
XGBoost    LSTM
(Transaction  (Weekly
  level)    time-series)
    │         │
    └────┬────┘
         ▼
   Hybrid Model
  (60% XGB + 40% LSTM)
         │
         ▼
  Streamlit Dashboard
  (5 pages · Live filters · Forecast engine)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total rows | 15,503 |
| Columns | 47 |
| Date range | Jan 2024 – Dec 2025 |
| Distributors | 15 (across Tamil Nadu) |
| SKUs | 30 |
| Beats per distributor | 6 |
| SBUs | Personal Care · Home Care · Home Essentials |
| Total outlets | 3,045 |

### Beat (Route) Structure

| Beat Type | DS Route | Outlet Types Served | Outlets |
|---|---|---|---|
| SWD | SWD Route | Town Wholesale, Rural Wholesale only | 20–22 |
| GR2 | GR2 Route | Retail, Medical, Premium Grocery (not SWD outlets) | 35–38 |
| ROC-A / ROC-B | ROC Route | All outlet types (outer market) | 25–28 each |
| 2W | 2W Route | Grocery, Medical, Wholesale (local market) | 38–42 |
| CMN | Common Pace | All outlet types (low-potential towns) | 50–52 |

---

## 🤖 Models

### XGBoost (Primary Model)
- **Level:** Transaction-level revenue prediction
- **Features:** 41 engineered features including lag-1/3/7/14/21/30, rolling averages, cyclic month encoding, zone/tier/beat encodings, SKU pricing signals
- **R²:** 0.9988

### LSTM (Deep Learning)
- **Level:** Weekly time-series demand forecasting
- **Architecture:** 2-layer LSTM (64 → 32 units) with Dropout (0.20)
- **Input:** 8-week sliding windows
- **Output:** Next-week NSV prediction

### Hybrid Model
- **Combination:** 60% XGBoost + 40% LSTM (weighted average, aligned at weekly level)
- Captures both **business relationships** (XGBoost) and **sequential patterns** (LSTM)

### Model Comparison

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | ₹XXX | ₹XXX | 0.9988 |
| LSTM | ₹XXX | ₹XXX | X.XXXX |
| **Hybrid** | **₹XXX** | **₹XXX** | **X.XXXX** |

*(Fill after training — run `python train_all_models.py`)*

---

## 📱 Dashboard Pages

| Page | Features |
|---|---|
| 🏠 Executive Dashboard | 6 KPIs with YoY delta · Monthly trend · Beat bar · Zone×Tier heatmap · SBU donut |
| 📈 Sales Forecast | Historical + forecast line · Confidence band · XGBoost-calibrated projection · YoY extension · Action Plan table |
| 🏢 Distributor Analysis | Revenue ranking · Achievement % · Beat×Distributor heatmap · Top 5 monthly trend |
| 📦 SKU & Category | Top 15 SKUs · Category + margin dual-axis · Discount vs margin scatter · Slow movers |
| 🚚 Delivery Intelligence | Lead time · On-time rate · Beat delivery rate · Outstanding payments |
| 🤖 Model Comparison | RMSE/MAE/R² comparison · Actual vs predicted · Feature importance · LSTM loss curve |

---

## 🔮 Forecast Action Plan

The forecast engine generates **bi-monthly review checkpoints (2, 4, 6, 8, 10, 12 months)** with automated suggested actions:

| Growth Signal | Suggested Action |
|---|---|
| ≥ 5% | ✅ On track — maintain current plan |
| 0–5% | 🟡 Flat — consider light SKU promo |
| 0 to -8% | ⚠️ De-growth — plan distributor scheme |
| Below -8% | 🔴 Significant de-growth — review beat coverage & SKU mix |

Available at SBU, Category, and SKU level with individual growth % and suggested actions.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Machine Learning | XGBoost 2.0, Scikit-learn |
| Deep Learning | TensorFlow 2.16, Keras |
| Dashboard | Streamlit 1.58 |
| Visualisation | Plotly 6.8 |
| Data | Pandas, NumPy |
| Model Storage | Joblib |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Rokeshkannan/FMCG-SALES-FORECASTING.git
cd FMCG-SALES-FORECASTING

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Add your dataset
# Place Dataset.csv in the project root

# 5. Train all models (XGBoost + LSTM + Hybrid)
python train_all_models.py

# 6. Launch dashboard
python -m streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
FMCG-SALES-FORECASTING/
├── dashboard/
│   └── app.py                  ← Streamlit dashboard (6 pages)
├── models/
│   ├── model_metrics.json      ← RMSE/MAE/R² all models
│   ├── feature_importance.csv  ← XGBoost feature rankings
│   ├── predictions_comparison.csv
│   ├── lstm_loss_history.csv
│   └── weekly_series.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_XGBoost_Model.ipynb
│   ├── 05_LSTM_Model.ipynb
│   └── 06_Hybrid_Model.ipynb
├── train_all_models.py         ← One-click training script
├── Dataset.csv                 ← FMCG sales data 2024–2025
├── requirements.txt
└── README.md
```

---

## 🎓 About

Built as a **career-defining portfolio project** at **Imarticus Learning, Bangalore** — demonstrating end-to-end data science skills across:
- Real-world FMCG domain knowledge
- Advanced feature engineering
- Machine Learning & Deep Learning
- Production-quality dashboard development

---

## 👨‍💻 Author

**Rocky** — Data Science Student, Imarticus Learning Bangalore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Rokeshkannan)

---

*⭐ Star this repo if you found it useful!*
