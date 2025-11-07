import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

DATA_PATH = Path("data/intraday_5min.parquet")
BASE_COLS = ["EXCHANGE", "TICKER", "PER", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "TYPE"]
TECH_COLS = [
    "SMA12", "SMA26", "EMA12", "EMA26", "MACD", "MACD_SIGNAL", "MACD_HIST",
    "BB_UPPER", "BB_LOWER", "ATR14", "NATR14", "RSI14", "STOCH_K14", "STOCH_D3",
    "OBV", "MFI14", "CMF20", "VWAP", "SIG_MA_LONG", "SIG_MA_SHORT", "SIG_MACD_LONG",
    "SIG_MACD_SHORT", "SIG_RSI_LONG", "SIG_RSI_SHORT", "SIG_BB_BREAK_UP",
    "SIG_BB_BREAK_DOWN", "SIG_BB_REVERT_LONG", "SIG_BB_REVERT_SHORT",
    "SIG_VWAP_ABOVE", "SIG_VWAP_BELOW", "SIG_VOL_SPIKE",
]
DROP_FOR_MODEL = ["EXCHANGE", "TICKER", "PER", "TYPE"]

def load_data(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    elif "index" in df.columns:
        df["index"] = pd.to_datetime(df["index"])
        df = df.sort_values("index").rename(columns={"index": "datetime"})
    else:
        df = df.sort_index()
    return df

def time_based_split(df: pd.DataFrame, train_frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * train_frac)
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    return train, test

def build_xy(df: pd.DataFrame, task: str, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    if task == "cls":
        target_col = f"CLS_UP{horizon}"
    else:
        target_col = f"RET_FWD{horizon}"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe.")
    avail_feats = [c for c in TECH_COLS if c in df.columns]
    ohlcv = [c for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"] if c in df.columns]
    feats = avail_feats + ohlcv
    X = df[feats].copy()
    y = df[target_col].copy()
    return X, y

def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def run_classification_models(X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, float]]:
    results = {}
    logreg = LogisticRegression(max_iter=500, n_jobs=-1)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_prob = logreg.predict_proba(X_test)[:, 1]
    results["logreg"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=20, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    results["mlp"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }
    return results

def run_regression_models(X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, float]]:
    results = {}
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred = lin.predict(X_test)
    results["linear"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["random_forest"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=20, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    results["mlp"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    return results