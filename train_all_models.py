"""
================================================================
  FMCG Sales Forecasting — Complete Model Training Script
  File : train_all_models.py
  Place: project root  (same level as data/ and models/ folders)
  Run  : python train_all_models.py
================================================================
Trains XGBoost + LSTM + Hybrid, saves all artefacts to models/
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── PATHS ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "Dataset.csv")          # ← change if needed
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts", "model_results")

for d in [MODELS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("  FMCG MODEL TRAINING PIPELINE")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & PREPROCESS
# ════════════════════════════════════════════════════════════════
print("\n[1/7] Loading and preprocessing dataset...")

df = pd.read_csv(DATA_FILE, parse_dates=["Order_Date"])
df["Delivery_Date"]     = pd.to_datetime(df["Delivery_Date"],     errors="coerce")
df["Expected_Delivery"] = pd.to_datetime(df["Expected_Delivery"], errors="coerce")
df_sales = df[df["Is_Return"] == "No"].copy()
df_sales = df_sales.sort_values(["Distributor_Code", "Order_Date"]).reset_index(drop=True)

print(f"  Sales rows loaded  : {len(df_sales):,}")

# Date features
df_sales["Year"]         = df_sales["Order_Date"].dt.year
df_sales["Month_Number"] = df_sales["Order_Date"].dt.month
df_sales["Quarter_Num"]  = df_sales["Quarter"].map({"Q1":1,"Q2":2,"Q3":3,"Q4":4})
df_sales["Week_Num"]     = df_sales["Order_Date"].dt.isocalendar().week.astype(int)
df_sales["Day_Num"]      = df_sales["Order_Date"].dt.dayofweek
df_sales["Month_Sin"]    = np.sin(2 * np.pi * df_sales["Month_Number"] / 12)
df_sales["Month_Cos"]    = np.cos(2 * np.pi * df_sales["Month_Number"] / 12)

# Encode categoricals
ENCODE_COLS = ["Zone","Tier","City","Beat_Name","DS_Type","Channel_Type",
               "Outlet_Type","SBU","Category","SKU_Code",
               "Payment_Mode","Payment_Status","Delivery_Status"]
label_encoders = {}
for col in ENCODE_COLS:
    if col in df_sales.columns:
        le = LabelEncoder()
        df_sales[col + "_enc"] = le.fit_transform(df_sales[col].astype(str))
        label_encoders[col] = le

joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))

# Lag & rolling features
TARGET = "Net_Sales_Value"
for lag in [1, 3, 7, 14, 21, 30]:
    df_sales[f"Sales_Lag_{lag}"] = (
        df_sales.groupby("Distributor_Code")[TARGET].shift(lag))
for w in [3, 7, 14, 21, 30]:
    df_sales[f"Rolling_{w}"] = (
        df_sales.groupby("Distributor_Code")[TARGET]
                .transform(lambda x: x.rolling(w).mean()))

# Business features
df_sales["Target_Gap"]            = df_sales["Target_Value"] - df_sales[TARGET]
df_sales["Achievement_Ratio"]     = np.where(
    df_sales["Target_Value"] > 0,
    df_sales[TARGET] / df_sales["Target_Value"], 0)
df_sales["Distributor_Margin_Pct"] = np.where(
    df_sales["Distributor_Price"] > 0,
    (df_sales["Distributor_Price"] - df_sales["Cost_Price"])
    / df_sales["Distributor_Price"] * 100, 0)

df_sales = df_sales.dropna().reset_index(drop=True)
print(f"  After dropna       : {len(df_sales):,} rows")

# ML feature list
ML_FEATURES = [c for c in [
    "Year","Month_Number","Quarter_Num","Week_Num","Day_Num",
    "Month_Sin","Month_Cos",
    "Zone_enc","Tier_enc","City_enc","Beat_Name_enc","DS_Type_enc",
    "Channel_Type_enc","Outlet_Type_enc",
    "SBU_enc","Category_enc","SKU_Code_enc",
    "MRP","Distributor_Price","Cost_Price",
    "Discount_Pct","Gross_Margin_Pct","Achievement_Pct",
    "Distributor_Margin_Pct","Target_Value","Target_Gap","Achievement_Ratio",
    "Payment_Mode_enc","Payment_Status_enc","Delivery_Status_enc",
    "Sales_Lag_1","Sales_Lag_3","Sales_Lag_7",
    "Sales_Lag_14","Sales_Lag_21","Sales_Lag_30",
    "Rolling_3","Rolling_7","Rolling_14","Rolling_21","Rolling_30",
] if c in df_sales.columns]

X = df_sales[ML_FEATURES]
y = df_sales[TARGET]

# Chronological 80/20 split
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler_X   = MinMaxScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

joblib.dump(scaler_X, os.path.join(MODELS_DIR, "scaler_X.pkl"))

print(f"  Train rows         : {len(X_train):,}")
print(f"  Test rows          : {len(X_test):,}")
print(f"  Features           : {len(ML_FEATURES)}")

# ════════════════════════════════════════════════════════════════
# STEP 2 — XGBOOST
# ════════════════════════════════════════════════════════════════
print("\n[2/7] Training XGBoost model...")

from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.04,
    subsample=0.80, colsample_bytree=0.80,
    min_child_weight=3, gamma=0.1,
    random_state=42, n_jobs=-1
)
model_xgb.fit(X_train_sc, y_train,
              eval_set=[(X_test_sc, y_test)], verbose=100)

preds_xgb_raw = model_xgb.predict(X_test_sc)

rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb_raw))
mae_xgb  = mean_absolute_error(y_test, preds_xgb_raw)
r2_xgb   = r2_score(y_test, preds_xgb_raw)

print(f"  XGBoost RMSE : ₹{rmse_xgb:,.2f}")
print(f"  XGBoost MAE  : ₹{mae_xgb:,.2f}")
print(f"  XGBoost R²   : {r2_xgb:.4f}")

joblib.dump(model_xgb, os.path.join(MODELS_DIR, "xgb_model.pkl"))

# Save feature importance
feat_imp = pd.DataFrame({
    "Feature": ML_FEATURES,
    "Importance": model_xgb.feature_importances_
}).sort_values("Importance", ascending=False)
feat_imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)

# Save XGBoost test predictions (row level)
xgb_test_df = df_sales.iloc[split_idx:][["Order_Date","Distributor_Name",
                                          "Beat_Key","SBU","Category"]].copy()
xgb_test_df["Actual_NSV"] = y_test.values
xgb_test_df["XGB_Pred"]   = preds_xgb_raw
xgb_test_df.to_csv(os.path.join(MODELS_DIR, "xgb_test_predictions.csv"), index=False)

print("  XGBoost model saved → models/xgb_model.pkl")

# ════════════════════════════════════════════════════════════════
# STEP 3 — WEEKLY SERIES FOR LSTM
# ════════════════════════════════════════════════════════════════
print("\n[3/7] Building weekly time-series for LSTM...")

df_full_sales = df[df["Is_Return"] == "No"].copy()
df_full_sales["Order_Date"]  = pd.to_datetime(df_full_sales["Order_Date"])
df_full_sales["Week_Start"]  = (df_full_sales["Order_Date"]
    - pd.to_timedelta(df_full_sales["Order_Date"].dt.weekday, unit="D"))

weekly = (df_full_sales.groupby("Week_Start")[TARGET]
          .sum().reset_index().sort_values("Week_Start").reset_index(drop=True))
weekly.columns = ["Week_Start", "NSV"]

print(f"  Weekly data points : {len(weekly)}")
print(f"  NSV range          : ₹{weekly['NSV'].min():,.0f} – ₹{weekly['NSV'].max():,.0f}")

# Scale
values   = weekly["NSV"].values.reshape(-1, 1)
scaler_y = MinMaxScaler()
scaled   = scaler_y.fit_transform(values)
joblib.dump(scaler_y, os.path.join(MODELS_DIR, "scaler_y.pkl"))

# Save weekly series for dashboard use
weekly.to_csv(os.path.join(MODELS_DIR, "weekly_series.csv"), index=False)

# Sliding windows (input=8 weeks → predict week 9)
WINDOW = 8
X_seq, y_seq = [], []
for i in range(WINDOW, len(scaled)):
    X_seq.append(scaled[i-WINDOW:i, 0])
    y_seq.append(scaled[i, 0])

X_seq = np.array(X_seq).reshape(-1, WINDOW, 1)
y_seq = np.array(y_seq)

lstm_split = int(len(X_seq) * 0.80)
X_tr, X_te = X_seq[:lstm_split], X_seq[lstm_split:]
y_tr, y_te = y_seq[:lstm_split], y_seq[lstm_split:]

print(f"  LSTM sequences     : {len(X_seq)} (train={len(X_tr)}, test={len(X_te)})")

# ════════════════════════════════════════════════════════════════
# STEP 4 — LSTM
# ════════════════════════════════════════════════════════════════
print("\n[4/7] Training LSTM model...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")

model_lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.20),
    LSTM(32, return_sequences=False),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1)
])
model_lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_lstm.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(lstm_path, save_best_only=True, verbose=0)
]
history = model_lstm.fit(
    X_tr, y_tr,
    epochs=150, batch_size=8,
    validation_data=(X_te, y_te),
    callbacks=callbacks, verbose=1
)

# Evaluate LSTM
pred_scaled  = model_lstm.predict(X_te)
preds_lstm   = scaler_y.inverse_transform(pred_scaled).flatten()
actual_lstm  = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()
weeks_test   = weekly["Week_Start"].iloc[lstm_split + WINDOW:].reset_index(drop=True)

rmse_lstm = np.sqrt(mean_squared_error(actual_lstm, preds_lstm))
mae_lstm  = mean_absolute_error(actual_lstm, preds_lstm)
r2_lstm   = r2_score(actual_lstm, preds_lstm)

print(f"\n  LSTM RMSE : ₹{rmse_lstm:,.2f}")
print(f"  LSTM MAE  : ₹{mae_lstm:,.2f}")
print(f"  LSTM R²   : {r2_lstm:.4f}")

# Save LSTM loss history
loss_df = pd.DataFrame({
    "Epoch":     range(1, len(history.history["loss"]) + 1),
    "Train_Loss": history.history["loss"],
    "Val_Loss":   history.history["val_loss"]
})
loss_df.to_csv(os.path.join(MODELS_DIR, "lstm_loss_history.csv"), index=False)

# Save LSTM test predictions
lstm_test_df = pd.DataFrame({
    "Week_Start":  weeks_test.values[:len(actual_lstm)],
    "Actual_NSV":  actual_lstm,
    "LSTM_Pred":   preds_lstm
})
lstm_test_df.to_csv(os.path.join(MODELS_DIR, "lstm_test_predictions.csv"), index=False)

print("  LSTM model saved → models/lstm_model.h5")

# ════════════════════════════════════════════════════════════════
# STEP 5 — HYBRID MODEL
# ════════════════════════════════════════════════════════════════
print("\n[5/7] Building hybrid model...")

# Aggregate XGBoost predictions to weekly to match LSTM level
xgb_weekly_df = xgb_test_df.copy()
xgb_weekly_df["Order_Date"] = pd.to_datetime(xgb_weekly_df["Order_Date"])
xgb_weekly_df["Week_Start"] = (xgb_weekly_df["Order_Date"]
    - pd.to_timedelta(xgb_weekly_df["Order_Date"].dt.weekday, unit="D"))

xgb_weekly = (xgb_weekly_df.groupby("Week_Start")
              .agg(Actual_NSV=("Actual_NSV","sum"),
                   XGB_Pred  =("XGB_Pred","sum"))
              .reset_index().sort_values("Week_Start"))

# Merge with LSTM predictions on matching weeks
lstm_test_df["Week_Start"] = pd.to_datetime(lstm_test_df["Week_Start"])
merged = pd.merge(xgb_weekly, lstm_test_df[["Week_Start","LSTM_Pred"]],
                  on="Week_Start", how="inner")

print(f"  Aligned weeks for comparison: {len(merged)}")

if len(merged) >= 5:
    W_XGB, W_LSTM = 0.60, 0.40
    merged["Hybrid_Pred"] = W_XGB * merged["XGB_Pred"] + W_LSTM * merged["LSTM_Pred"]

    rmse_hyb = np.sqrt(mean_squared_error(merged["Actual_NSV"], merged["Hybrid_Pred"]))
    mae_hyb  = mean_absolute_error(merged["Actual_NSV"], merged["Hybrid_Pred"])
    r2_hyb   = r2_score(merged["Actual_NSV"], merged["Hybrid_Pred"])

    # Also compute XGB and LSTM metrics at this same weekly level
    rmse_xgb_w = np.sqrt(mean_squared_error(merged["Actual_NSV"], merged["XGB_Pred"]))
    mae_xgb_w  = mean_absolute_error(merged["Actual_NSV"], merged["XGB_Pred"])
    r2_xgb_w   = r2_score(merged["Actual_NSV"], merged["XGB_Pred"])

    rmse_lstm_w = np.sqrt(mean_squared_error(merged["Actual_NSV"], merged["LSTM_Pred"]))
    mae_lstm_w  = mean_absolute_error(merged["Actual_NSV"], merged["LSTM_Pred"])
    r2_lstm_w   = r2_score(merged["Actual_NSV"], merged["LSTM_Pred"])

    print(f"  Hybrid RMSE : ₹{rmse_hyb:,.2f}")
    print(f"  Hybrid MAE  : ₹{mae_hyb:,.2f}")
    print(f"  Hybrid R²   : {r2_hyb:.4f}")

    merged.to_csv(os.path.join(MODELS_DIR, "predictions_comparison.csv"), index=False)

else:
    print("  ⚠ Not enough overlapping weeks. Using transaction-level XGB metrics.")
    rmse_xgb_w, mae_xgb_w, r2_xgb_w   = rmse_xgb, mae_xgb, r2_xgb
    rmse_lstm_w, mae_lstm_w, r2_lstm_w  = rmse_lstm, mae_lstm, r2_lstm
    rmse_hyb = (rmse_xgb + rmse_lstm) / 2
    mae_hyb  = (mae_xgb  + mae_lstm)  / 2
    r2_hyb   = (r2_xgb   + r2_lstm)   / 2

# ════════════════════════════════════════════════════════════════
# STEP 6 — SAVE METRICS JSON
# ════════════════════════════════════════════════════════════════
print("\n[6/7] Saving model metrics...")

metrics = {
    "XGBoost": {
        "RMSE": round(float(rmse_xgb_w), 2),
        "MAE":  round(float(mae_xgb_w),  2),
        "R2":   round(float(r2_xgb_w),   4),
        "level": "weekly_aggregated"
    },
    "LSTM": {
        "RMSE": round(float(rmse_lstm_w), 2),
        "MAE":  round(float(mae_lstm_w),  2),
        "R2":   round(float(r2_lstm_w),   4),
        "level": "weekly"
    },
    "Hybrid": {
        "RMSE": round(float(rmse_hyb), 2),
        "MAE":  round(float(mae_hyb),  2),
        "R2":   round(float(r2_hyb),   4),
        "level": "weekly",
        "weights": {"XGBoost": 0.60, "LSTM": 0.40}
    },
    "training_info": {
        "dataset_rows":     int(len(df_sales)),
        "train_rows":       int(len(X_train)),
        "test_rows":        int(len(X_test)),
        "features_count":   int(len(ML_FEATURES)),
        "weekly_points":    int(len(weekly)),
        "lstm_window":      int(WINDOW),
        "lstm_train_seqs":  int(len(X_tr)),
        "lstm_test_seqs":   int(len(X_te)),
        "split_ratio":      0.80
    }
}

with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"  model_metrics.json saved")
print(f"\n  ┌─────────────┬───────────┬──────────┬────────┐")
print(f"  │ Model       │   RMSE    │   MAE    │   R²   │")
print(f"  ├─────────────┼───────────┼──────────┼────────┤")
print(f"  │ XGBoost     │ ₹{rmse_xgb_w:>8,.0f}│ ₹{mae_xgb_w:>7,.0f}│ {r2_xgb_w:>6.4f}│")
print(f"  │ LSTM        │ ₹{rmse_lstm_w:>8,.0f}│ ₹{mae_lstm_w:>7,.0f}│ {r2_lstm_w:>6.4f}│")
print(f"  │ Hybrid      │ ₹{rmse_hyb:>8,.0f}│ ₹{mae_hyb:>7,.0f}│ {r2_hyb:>6.4f}│")
print(f"  └─────────────┴───────────┴──────────┴────────┘")

# ════════════════════════════════════════════════════════════════
# STEP 7 — SAVE CHARTS
# ════════════════════════════════════════════════════════════════
print("\n[7/7] Saving charts...")

plt.style.use("dark_background")

# Feature importance chart
fig, ax = plt.subplots(figsize=(10, 8))
top15 = feat_imp.head(15)
bars = ax.barh(top15["Feature"][::-1], top15["Importance"][::-1],
               color="#3B82F6", edgecolor="none", alpha=0.85)
ax.set_title("XGBoost — Top 15 Feature Importances", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "xgb_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# LSTM loss curve
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(loss_df["Epoch"], loss_df["Train_Loss"], color="#3B82F6", lw=2, label="Train Loss")
ax.plot(loss_df["Epoch"], loss_df["Val_Loss"],   color="#EF4444", lw=2, linestyle="--", label="Val Loss")
ax.set_title("LSTM — Training vs Validation Loss", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "lstm_loss_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# Actual vs Predicted — all 3 models (weekly)
if os.path.exists(os.path.join(MODELS_DIR, "predictions_comparison.csv")):
    cmp = pd.read_csv(os.path.join(MODELS_DIR, "predictions_comparison.csv"),
                      parse_dates=["Week_Start"])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cmp["Week_Start"], cmp["Actual_NSV"]/1e3,  color="#F1F5F9", lw=2.5, label="Actual")
    ax.plot(cmp["Week_Start"], cmp["XGB_Pred"]/1e3,    color="#3B82F6", lw=2,   label="XGBoost", linestyle="--")
    ax.plot(cmp["Week_Start"], cmp["LSTM_Pred"]/1e3,   color="#EF4444", lw=2,   label="LSTM",    linestyle=":")
    ax.plot(cmp["Week_Start"], cmp["Hybrid_Pred"]/1e3, color="#10B981", lw=2,   label="Hybrid",  linestyle="-.")
    ax.set_title("Model Comparison — Weekly NSV (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weekly NSV (₹K)"); ax.legend(); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "model_comparison_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()

print("  Charts saved → outputs/charts/model_results/")

# ════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TRAINING COMPLETE — Files saved to models/")
print("=" * 60)
print("""
  models/
  ├── xgb_model.pkl              ← XGBoost model
  ├── lstm_model.h5              ← LSTM model
  ├── scaler_X.pkl               ← Feature scaler
  ├── scaler_y.pkl               ← Weekly NSV scaler
  ├── label_encoders.pkl         ← All category encoders
  ├── model_metrics.json         ← RMSE/MAE/R² all models
  ├── feature_importance.csv     ← XGBoost feature ranks
  ├── predictions_comparison.csv ← Weekly actual vs all models
  ├── xgb_test_predictions.csv   ← Row-level XGB predictions
  ├── lstm_test_predictions.csv  ← Weekly LSTM predictions
  ├── lstm_loss_history.csv      ← Training loss per epoch
  └── weekly_series.csv          ← Full weekly NSV series

  Next step: streamlit run dashboard/app.py
""")
