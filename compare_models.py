
# compare_models.py : 3 anomaly models ka same data pe comparison

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from simulator import generate_stream
from features import add_features, FEATURES
from config import ALARM_RISK, GAS_THRESHOLD

# ---- data: training, calibration, test (teeno alag seed) ----
train = add_features(generate_stream(5000, seed=0))   # model sirf isse seekhega (normal)
calib = add_features(generate_stream(5000, seed=1))   # risk scale (lo/hi) isse banega (normal, unseen)

leaks = [(300, 150), (900, 150), (1500, 150), (2100, 150), (2600, 150)]
spikes = [100, 500, 700, 1200, 1300, 1800, 2000, 2400, 2800]
raw_test = generate_stream(3000, leaks, spikes, seed=123)
test = add_features(raw_test)
y = raw_test["label"].values

scaler = StandardScaler().fit(train[FEATURES])
X_train = scaler.transform(train[FEATURES])
X_calib = scaler.transform(calib[FEATURES])
X_test = scaler.transform(test[FEATURES])

models = {
    "IsolationForest": IsolationForest(n_estimators=200, random_state=42),
    "OneClassSVM": OneClassSVM(nu=0.01, gamma="scale"),
    "LOF": LocalOutlierFactor(n_neighbors=35, novelty=True),
}


def to_risk(model, X, lo, hi):
    s = -model.score_samples(X)
    r = np.clip(50 * (s - lo) / (hi - lo), 0, 100)
    return pd.Series(r).rolling(5, min_periods=1).mean().values


def metrics(pred):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return prec, rec, f1, fp / (fp + tn)


def latency(pred):
    out = []
    for start, dur in leaks:
        hit = np.where(pred[start:start + dur])[0]
        out.append(hit[0] if len(hit) else np.nan)
    out = np.array(out, dtype=float)
    return np.nanmean(out) if not np.all(np.isnan(out)) else np.nan, int(np.isnan(out).sum())


rows = []

# ---- baseline ----
base = (raw_test["gas"] > GAS_THRESHOLD).values
p, r, f, fpr = metrics(base)
lat, miss = latency(base)
rows.append(["Fixed threshold", p, r, f, fpr, lat, miss, int(base[spikes].sum()), 0.0])

# ---- 3 AI models ----
for name, model in models.items():
    model.fit(X_train)
    s_cal = -model.score_samples(X_calib)
    lo, hi = float(np.median(s_cal)), float(np.percentile(s_cal, 99.9))

    t0 = time.perf_counter()
    risk = to_risk(model, X_test, lo, hi)
    ms = (time.perf_counter() - t0) / len(X_test) * 1000

    pred = risk >= ALARM_RISK
    p, r, f, fpr = metrics(pred)
    lat, miss = latency(pred)
    rows.append([name, p, r, f, fpr, lat, miss, int(pred[spikes].sum()), ms])

res = pd.DataFrame(rows, columns=[
    "Model", "Precision", "Recall", "F1", "FPR",
    "AvgLatency(s)", "MissedLeaks", "SpikeAlarms", "ms/reading"])
print(res.round(4).to_string(index=False))
res.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")
