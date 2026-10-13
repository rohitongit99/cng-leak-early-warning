# final_eval.py : AI vs fixed threshold (alag-alag values + persistence), 10 seeds pe

import numpy as np
import pandas as pd

from simulator import generate_stream
from risk import load, risk_from_df
from config import ALARM_RISK

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)

bundle = load()   # wahi saved model jo dashboard use karta hai

leaks = [(300, 150), (900, 150), (1500, 150), (2100, 150), (2600, 150)]
spikes = [100, 500, 700, 1200, 1300, 1800, 2000, 2400, 2800]
SEEDS = list(range(201, 211))   # 10 unseen test seeds
THRESHOLDS = [10, 20, 30, 50]


def persist(mask, n=3):
    """Alarm tab jab mask lagatar n readings True rahe."""
    s = pd.Series(mask.astype(int)).rolling(n).min().fillna(0)
    return (s == 1).values


methods = {}
for T in THRESHOLDS:
    methods[f"Threshold {T} ppm"] = lambda df, T=T: (df["gas"] > T).values
for T in THRESHOLDS:
    methods[f"Threshold {T} ppm + 3s persist"] = lambda df, T=T: persist((df["gas"] > T).values)
methods["AI (Isolation Forest)"] = lambda df: (risk_from_df(df, bundle) >= ALARM_RISK).values


def evaluate(pred, y):
    tp = (pred & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    fn = (~pred & (y == 1)).sum()
    tn = (~pred & (y == 0)).sum()
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn)

    lat = []
    for start, dur in leaks:
        hit = np.where(pred[start:start + dur])[0]
        lat.append(hit[0] if len(hit) else np.nan)
    lat = np.array(lat, dtype=float)
    avg_lat = np.nanmean(lat) if not np.all(np.isnan(lat)) else np.nan
    return dict(F1=f1, Precision=prec, Recall=rec, FPR=fpr, Latency=avg_lat,
                Missed=int(np.isnan(lat).sum()), SpikeAlarms=int(pred[spikes].sum()))


results = {name: [] for name in methods}
for sd in SEEDS:
    raw = generate_stream(3000, leaks, spikes, seed=sd)
    y = raw["label"].values
    for name, fn in methods.items():
        results[name].append(evaluate(fn(raw), y))
    print("seed done:", sd, flush=True)


def pm(vals, d=3):
    v = np.array(vals, dtype=float)
    if np.all(np.isnan(v)):
        return "n/a"
    return f"{np.nanmean(v):.{d}f} +- {np.nanstd(v):.{d}f}"


rows = []
for name, lst in results.items():
    rows.append([
        name,
        pm([r["F1"] for r in lst]),
        pm([r["Precision"] for r in lst]),
        pm([r["Recall"] for r in lst]),
        pm([r["FPR"] for r in lst], 4),
        pm([r["Latency"] for r in lst], 1),
        pm([r["Missed"] for r in lst], 1),
        pm([r["SpikeAlarms"] for r in lst], 1),
    ])

res = pd.DataFrame(rows, columns=[
    "Method", "F1", "Precision", "Recall", "FPR",
    "Latency_s", "Missed(of 5)", "SpikeAlarms(of 9)"])
print()
print(res.to_string(index=False))
res.to_csv("final_results.csv", index=False)
print("\nSaved: final_results.csv")