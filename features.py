# features.py : raw readings se naye useful columns (features) banata hai

FEATURES = ["gas", "pressure", "temp", "rpm", "speed",
            "gas_roc", "gas_med", "pressure_slope", "temp_slope"]


def add_features(df):
    d = df.copy()
    d["gas_roc"] = d["gas"].diff(5) / 5                          # gas badalne ki speed (rate of change)
    d["gas_med"] = d["gas"].rolling(10, min_periods=1).median()  # median: ek akeli spike ko ignore karta hai
    d["pressure_slope"] = d["pressure"].diff(10) / 10            # pressure ka trend
    d["temp_slope"] = d["temp"].diff(10) / 10                    # temperature ka trend
    return d.fillna(0)