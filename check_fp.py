# check_fp.py : AI ke false alarm kin rows pe aaye, ye dekhta hai

import numpy as np
from simulator import generate_stream
from risk import load, risk_from_df
from config import ALARM_RISK

bundle = load()

leaks = [(300, 150), (900, 150), (1500, 150), (2100, 150), (2600, 150)]
spikes = [100, 500, 700, 1200, 1300, 1800, 2000, 2400, 2800]
df = generate_stream(3000, leaks, spikes, seed=123)

y = df["label"].values
ai = (risk_from_df(df, bundle) >= ALARM_RISK).values

fp_rows = np.where(ai & (y == 0))[0]
print("Total AI false alarm rows:", len(fp_rows))
print("Row numbers:", fp_rows.tolist())