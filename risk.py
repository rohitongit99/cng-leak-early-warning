# risk.py : model ke output ko 0-100 risk score aur level mein badalta hai

import joblib
import numpy as np
import pandas as pd

from features import add_features, FEATURES


def load(path="models/iforest.joblib"):
    """train.py ne jo model save kiya tha, use wapas load karta hai."""
    return joblib.load(path)


def risk_from_df(df_raw, bundle, smooth=5):
    """Raw readings -> risk score (0-100), smooth kiya hua."""
    df = add_features(df_raw)
    X = bundle["scaler"].transform(df[FEATURES])
    s = -bundle["model"].score_samples(X)

    # median score -> 0, 99.9th percentile score -> 50, usse aage 100 tak
    r = 50 * (s - bundle["lo"]) / (bundle["hi"] - bundle["lo"])
    r = np.clip(r, 0, 100)

    # smoothing: pichhli 5 readings ka average, taaki ek akeli reading alarm na bane
    return pd.Series(r, index=df.index).rolling(smooth, min_periods=1).mean()


def level(risk):
    if risk < 25:
        return "SAFE"
    if risk < 50:
        return "WATCH"
    if risk < 75:
        return "WARNING"
    return "DANGER"