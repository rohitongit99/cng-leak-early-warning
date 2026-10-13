# evaluate.py : fixed-threshold baseline vs AI model ka comparison

import numpy as np
from simulator import generate_stream
from risk import load, risk_from_df
from config import GAS_THRESHOLD, ALARM_RISK

bundle = load()

# Test data: training se alag seed, to model ne ye data kabhi dekha nahi
leaks = [(300, 150), (900, 150), (1500, 150), (2100, 150), (2600, 150)]
spikes = [100, 500, 700, 1200, 1300, 1800, 2000, 2400, 2800]
df = generate_stream(3000, leaks, spikes, seed=123)

y = df["label"].values
ai = (risk_from_df(df, bundle) >= ALARM_RISK).values
base = (df["gas"] > GAS_THRESHOLD).values


def metrics(pred, name):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn)
    print(f"{name:10s} TP={tp} FP={fp} FN={fn} TN={tn} | "
          f"precision={prec:.2f} recall={rec:.2f} F1={f1:.2f} FPR={fpr:.4f}")


def latencies(pred):
    """Har leak shuru hone ke kitne second baad alarm aaya."""
    out = []
    for start, dur in leaks:
        hit = np.where(pred[start:start + dur])[0]
        out.append(int(hit[0]) if len(hit) else np.nan)
    return np.array(out, dtype=float)


metrics(base, "Threshold")
metrics(ai, "AI model")

lb, la = latencies(base), latencies(ai)
print()
print("Detection latency per leak (seconds):")
print("Threshold:", lb)
print("AI model :", la)
print("Average early-warning gain (s):", np.nanmean(lb - la))

print()
print("Spike pe alarm kisne diya (9 spikes mein se):")
print("Threshold:", int(base[spikes].sum()))
print("AI model :", int(ai[spikes].sum()))