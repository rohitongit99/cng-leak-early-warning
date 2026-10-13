# ablation.py : ek-ek feature group hata ke dekhna ki result kitna badalta hai

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from simulator import generate_stream
from features import add_features, FEATURES
from config import ALARM_RISK

leaks = [(300, 150), (900, 150), (1500, 150), (2100, 150), (2600, 150)]
spikes = [100, 500, 700, 1200, 1300, 1800, 2000, 2400, 2800]
TEST_SEEDS = [101, 102, 103, 104, 105]


def drop(*names):
    return [f for f in FEATURES if f not in names]


VARIANTS = {
    "All features (full)": FEATURES,
    "No pressure features": drop("pressure", "pressure_slope"),
    "No temperature features": drop("temp", "temp_slope"),
    "No rpm/speed": drop("rpm", "speed"),
    "No gas_roc": drop("gas_roc"),
    "No gas_med": drop("gas_med"),
    "No slope features": drop("pressure_slope", "temp_slope"),
    "Gas only (gas, gas_roc, gas_med)": ["gas", "gas_roc", "gas_med"],
}

train = add_features(generate_stream(5000, seed=0))
calib = add_features(generate_stream(5000, seed=1))

tests = []
for sd in TEST_SEEDS:
    raw = generate_stream(3000, leaks, spikes, seed=sd)
    tests.append((add_features(raw), raw["label"].values))


def run_variant(cols):
    scaler = StandardScaler().fit(train[cols])
    model = IsolationForest(n_estimators=200, random_state=42).fit(scaler.transform(train[cols]))

    s_cal = -model.score_samples(scaler.transform(calib[cols]))
    lo, hi = float(np.median(s_cal)), float(np.percentile(s_cal, 99.9))

    f1s, fprs, lats, misses, spk = [], [], [], [], []
    for test, y in tests:
        s = -model.score_samples(scaler.transform(test[cols]))
        r = np.clip(50 * (s - lo) / (hi - lo), 0, 100)
        pred = (pd.Series(r).rolling(5, min_periods=1).mean() >= ALARM_RISK).values

        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = (~pred & (y == 1)).sum()
        tn = (~pred & (y == 0)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        fprs.append(fp / (fp + tn))

        lat = []
        for start, dur in leaks:
            hit = np.where(pred[start:start + dur])[0]
            lat.append(hit[0] if len(hit) else np.nan)
        lat = np.array(lat, dtype=float)
        misses.append(int(np.isnan(lat).sum()))
        if not np.all(np.isnan(lat)):
            lats.append(np.nanmean(lat))
        spk.append(int(pred[spikes].sum()))

    return [np.mean(f1s), np.std(f1s), np.mean(fprs), np.mean(lats) if lats else np.nan,
            np.mean(misses), np.mean(spk)]


rows = []
for name, cols in VARIANTS.items():
    print("Running:", name, flush=True)
    rows.append([name, len(cols)] + run_variant(cols))

res = pd.DataFrame(rows, columns=[
    "Variant", "nFeatures", "F1_mean", "F1_std", "FPR_mean",
    "Latency_s", "MissedLeaks", "SpikeAlarms(of 9)"])
print()
print(res.round(4).to_string(index=False))
res.to_csv("ablation.csv", index=False)
print("\nSaved: ablation.csv")