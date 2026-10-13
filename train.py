# train.py : Isolation Forest model ko normal data pe train karta hai

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from simulator import generate_stream
from features import add_features, FEATURES

# 1) Sirf NORMAL data banao (koi leak nahi, koi spike nahi)
df = add_features(generate_stream(20000, seed=0))
print("Training data:", df.shape)

# 2) Scale karo: sab features ko ek jaisi range mein laao
scaler = StandardScaler().fit(df[FEATURES])
X = scaler.transform(df[FEATURES])

# 3) Isolation Forest train karo
model = IsolationForest(n_estimators=200, random_state=42).fit(X)

# 4) Risk score ke liye calibration
#    score_samples jitna negative, utna anomalous. Hum sign ulta karte hain
#    taaki bada number = zyada anomalous.
s = -model.score_samples(X)
bundle = dict(
    model=model,
    scaler=scaler,
    lo=float(np.median(s)),            # normal ka typical score  -> risk 0
    hi=float(np.percentile(s, 99.9)),  # normal ka lagbhag sabse bura -> risk 50
)

# 5) Save karo
os.makedirs("models", exist_ok=True)
joblib.dump(bundle, "models/iforest.joblib")

print("lo =", round(bundle["lo"], 4), " hi =", round(bundle["hi"], 4))
print("Model saved: models/iforest.joblib")