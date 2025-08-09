# streamlit_app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Optional imports guarded
HAVE_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    HAVE_PROPHET = False

HAVE_STATSMODELS = True
try:
    import statsmodels.api as sm
except Exception:
    HAVE_STATSMODELS = False

HAVE_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAVE_XGB = False

HAVE_TF = True
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    HAVE_TF = False

st.set_page_config(page_title="Sales Forecasting Lab", layout="wide")

# ------------------ Helpers ------------------ #
@st.cache_data(show_spinner=False)
def load_data(uploaded, fallback_path="data/train.csv"):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv(fallback_path)
    # Basic standardization
    if "date" not in df.columns or "sales" not in df.columns:
        raise ValueError("CSV must include at least 'date' and 'sales' columns.")
    df["date"] = pd.to_datetime(df["date"])
    # Friendly column names
    return df

def aggregate_level(df, level):
    if level == "Total (all stores & items)":
        out = df.groupby("date")["sales"].sum().reset_index()
    elif level == "Per Store (sum items)":
        if "store" not in df.columns:
            st.warning("No 'store' column found. Falling back to Total.")
            return df.groupby("date")["sales"].sum().reset_index()
        sel_store = st.sidebar.selectbox(
            "Choose store (for Per Store)", sorted(df["store"].unique())
        )
        out = df[df["store"] == sel_store].groupby("date")["sales"].sum().reset_index()
    else:  # Per Item
        if "item" not in df.columns:
            st.warning("No 'item' column found. Falling back to Total.")
            return df.groupby("date")["sales"].sum().reset_index()
        sel_item = st.sidebar.selectbox(
            "Choose item (for Per Item)", sorted(df["item"].unique())
        )
        out = df[df["item"] == sel_item].groupby("date")["sales"].sum().reset_index()
    out = out.sort_values("date")
    out = out.rename(columns={"date": "ds", "sales": "y"})
    return out

def time_split(df, split_date):
    train = df[df["ds"] < split_date].copy()
    test = df[df["ds"] >= split_date].copy()
    if len(test) == 0:
        st.error("Your split produces an empty test set. Pick an earlier split date.")
    return train, test

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def plot_test_only(test_ds, y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(test_ds, y_true, marker="o", label="Actual")
    ax.plot(test_ds, y_pred, marker="x", label="Forecast")
    ax.set_title(f"{title} — Test Period Only")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales"); ax.grid(True); ax.legend()
    st.pyplot(fig)

# ------------------ Models ------------------ #
def run_prophet(train, test):
    if not HAVE_PROPHET:
        st.error("prophet is not installed.")
        return None
    m = Prophet()
    m.fit(train)
    fut = m.make_future_dataframe(periods=len(test))
    fc = m.predict(fut)
    fc_test = fc.iloc[-len(test):]
    return fc_test["yhat"].values

def run_sarima(train, test, order=(1,1,1), seasonal_order=(1,1,1,7)):
    if not HAVE_STATSMODELS:
        st.error("statsmodels is not installed.")
        return None
    y_train = train["y"].values
    model = sm.tsa.statespace.SARIMAX(
        y_train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    res = model.fit(disp=False)
    y_pred = res.forecast(steps=len(test))  # ndarray
    return y_pred

def make_lag_features(df, lags=(1,7,14,28), rolls=(7,14,28)):
    tmp = df.copy()
    for L in lags:
        tmp[f"lag_{L}"] = tmp["y"].shift(L)
    for R in rolls:
        tmp[f"rollmean_{R}"] = tmp["y"].shift(1).rolling(R).mean()
    cal = pd.DataFrame({"ds": tmp["ds"]})
    cal["dow"] = cal["ds"].dt.dayofweek
    cal["dom"] = cal["ds"].dt.day
    cal["month"] = cal["ds"].dt.month
    cal["week"] = cal["ds"].dt.isocalendar().week.astype(int)
    tmp = tmp.join(cal[["dow","dom","month","week"]])
    tmp = tmp.dropna().reset_index(drop=True)
    return tmp

def run_xgb(train, test):
    if not HAVE_XGB:
        st.error("xgboost is not installed.")
        return None, None
    full = pd.concat([train, test], ignore_index=True)
    feat = make_lag_features(full)
    split_idx = feat[feat["ds"] < train["ds"].iloc[-1]].index.max()
    X_cols = [c for c in feat.columns if c not in ["ds","y"]]
    X_train, y_train = feat.loc[:split_idx, X_cols], feat.loc[:split_idx, "y"]
    X_test,  y_test  = feat.loc[split_idx+1:, X_cols], feat.loc[split_idx+1:, "y"]

    model = XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_ds = feat.loc[split_idx+1:, "ds"].values
    return test_ds, y_test.values, y_pred

def add_calendar_feats(df):
    out = df.copy()
    out["dow"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["dow_sin"] = np.sin(2*np.pi*out["dow"]/7)
    out["dow_cos"] = np.cos(2*np.pi*out["dow"]/7)
    out["m_sin"] = np.sin(2*np.pi*out["month"]/12)
    out["m_cos"] = np.cos(2*np.pi*out["month"]/12)
    return out

def run_lstm(train, test, lookback=90, horizon=7, epochs=25, batch_size=64):
    if not HAVE_TF:
        st.error("tensorflow is not installed.")
        return None
    # Build feature frame
    tr, te = train.copy(), test.copy()
    full = pd.concat([tr, te], ignore_index=True)
    full = add_calendar_feats(full)

    # log1p stabilize
    full["y"] = np.log1p(full["y"])

    # scale features on train only
    feat_cols = [c for c in full.columns if c not in ["ds"]]
    scaler = MinMaxScaler()
    split_idx = len(tr)
    full.iloc[:split_idx, 1:] = scaler.fit_transform(full.iloc[:split_idx, 1:])
    full.iloc[split_idx:, 1:] = scaler.transform(full.iloc[split_idx:, 1:])

    # make supervised multistep
    def make_supervised_multistep(df, lookback=90, horizon=7):
        cols = [c for c in df.columns if c != "ds"]
        values = df[cols].values
        X, Y = [], []
        for i in range(lookback, len(df)-horizon+1):
            X.append(values[i-lookback:i, :])
            Y.append(values[i:i+horizon, 0])  # first col is y
        return np.array(X), np.array(Y)

    X, Y = make_supervised_multistep(full, lookback, horizon)
    usable = full.iloc[lookback:]
    cut = split_idx - lookback
    X_train, Y_train = X[:cut], Y[:cut]
    X_test_all, Y_test_all = X[cut:], Y[cut:]

    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, X.shape[2])),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(64),
        keras.layers.Dense(horizon)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    cb = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
    ]
    model.fit(X_train, Y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cb)

    # roll weekly predictions across test
    steps = int(np.ceil(len(test) / horizon))
    preds = []
    start = 0
    for _ in range(steps):
        x_in = X_test_all[start:start+1]
        if x_in.shape[0] == 0: break
        p = model.predict(x_in, verbose=0)[0]
        preds.append(p); start += horizon
    y_pred_scaled = np.concatenate(preds)[:len(test)]
    y_pred = np.expm1(y_pred_scaled)  # invert log1p
    return y_pred

# ------------------ UI ------------------ #
st.title("🛒 Sales Forecasting Lab")
st.caption("Compare Prophet, SARIMA, XGBoost, and LSTM on your daily sales data.")

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload train.csv (optional)", type=["csv"])
    split = st.date_input("Train/Test split date", value=pd.to_datetime("2017-09-01"))
    level = st.selectbox("Aggregation", ["Total (all stores & items)", "Per Store (sum items)", "Per Item (single item)"])

    st.header("2) Models")
    use_prophet = st.checkbox("Prophet", True and HAVE_PROPHET)
    use_sarima  = st.checkbox("SARIMA", HAVE_STATSMODELS)
    use_xgb     = st.checkbox("XGBoost", HAVE_XGB)
    use_lstm    = st.checkbox("LSTM", HAVE_TF)

    st.header("LSTM Params")
    lookback = st.slider("Lookback (days)", 30, 180, 90, step=10)
    horizon  = st.slider("Direct horizon (days)", 7, 28, 7, step=7)
    epochs   = st.slider("Epochs", 5, 200, 25, step=5)
    batchsz  = st.selectbox("Batch size", [32, 64, 128], index=1)

    run_btn = st.button("Run Models")

# Load & prep
try:
    raw = load_data(up)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

series = aggregate_level(raw, level)
train, test = time_split(series, pd.to_datetime(split))

if not run_btn:
    st.info("Choose options in the sidebar and click **Run Models**.")
    st.stop()

if len(test) == 0 or len(train) == 0:
    st.error("Train/Test split invalid. Adjust the split date.")
    st.stop()

results = []
plots = []

# Prophet
if use_prophet:
    with st.spinner("Training Prophet..."):
        y_pred = run_prophet(train, test)
        if y_pred is not None:
            mae, rmse = evaluate(test["y"].values, y_pred)
            results.append(("Prophet", mae, rmse))
            plots.append(("Prophet", test["ds"].values, test["y"].values, y_pred))

# SARIMA
if use_sarima:
    with st.spinner("Training SARIMA..."):
        y_pred = run_sarima(train, test, order=(1,1,1), seasonal_order=(1,1,1,7))
        if y_pred is not None:
            mae, rmse = evaluate(test["y"].values, y_pred)
            results.append(("SARIMA(1,1,1)x(1,1,1,7)", mae, rmse))
            plots.append(("SARIMA", test["ds"].values, test["y"].values, y_pred))

# XGBoost
if use_xgb:
    with st.spinner("Training XGBoost..."):
        test_ds_xgb, y_true_xgb, y_pred_xgb = run_xgb(train, test)
        if y_pred_xgb is not None:
            mae, rmse = evaluate(y_true_xgb, y_pred_xgb)
            results.append(("XGBoost (lags+roll)", mae, rmse))
            plots.append(("XGBoost", test_ds_xgb, y_true_xgb, y_pred_xgb))

# LSTM
if use_lstm:
    with st.spinner("Training LSTM..."):
        y_pred = run_lstm(train, test, lookback=lookback, horizon=horizon, epochs=epochs, batch_size=batchsz)
        if y_pred is not None:
            mae, rmse = evaluate(test["y"].values, y_pred)
            results.append((f"LSTM (lb={lookback},h={horizon})", mae, rmse))
            plots.append(("LSTM", test["ds"].values, test["y"].values, y_pred))

# Results table
if results:
    res_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE"]).sort_values("RMSE")
    st.subheader("📊 Results")
    st.dataframe(res_df, use_container_width=True)

# Plots (test only)
if plots:
    st.subheader("📈 Test Period Plots")
    tabs = st.tabs([name for name, *_ in plots])
    for tab, (name, ds, y_true, y_pred) in zip(tabs, plots):
        with tab:
            plot_test_only(ds, y_true, y_pred, name)

with st.expander("Advanced / Notes"):
    st.markdown("""
- Prophet handles yearly/weekly seasonality well out‑of‑the‑box.
- SARIMA uses weekly seasonality `m=7`. Try `(2,1,1,7)` or add yearly seasonality for longer data.
- XGBoost uses lag & rolling mean features + calendar encodings. Add holidays/promotions for gains.
- LSTM is set to direct **multi‑step** prediction (predict `horizon` days at once) with calendar features and log‑stabilized target.
- All plots show **test period only** (actual vs forecast).
    """)
