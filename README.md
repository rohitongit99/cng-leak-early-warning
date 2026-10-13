# 🔥 AI-Based CNG Leakage Early-Warning System

A software-only research prototype that tests whether **multivariate anomaly detection** can warn about a CNG leak earlier than a fixed gas threshold, and shows the warning on a live car-style dashboard.

> ⚠️ **Research prototype on simulated data. Not a certified automotive safety device.**


## Why this project
Most vehicle gas alarms use one fixed limit. Slow leaks are caught late, and a one-second sensor glitch can trigger a false alarm. This project studies an Isolation Forest on time-based features as an alternative, and compares it honestly against threshold baselines.

## Features
- 🧪 Seed-controlled **simulator** (normal driving, noise, sensor glitches, gradual leaks)
- 📈 **Feature engineering**: gas rate-of-change, rolling median, pressure and temperature slopes
- 🌲 **Isolation Forest** trained on normal data only (unsupervised)
- 🎯 **Risk score 0-100** with 4 levels: SAFE, WATCH, WARNING, DANGER
- 🚗 **Streamlit dashboard**: gauges, warning lights, odometer, live charts
- 🚨 **Latching alarm** with selectable sound, stops only when the driver switches to petrol or stops the vehicle
- 📊 Evaluation scripts: model comparison, ablation study, tuned-threshold comparison over 10 seeds

## Pipeline
```
Simulator -> Feature engineering -> Isolation Forest -> Risk score -> Dashboard + alarm
```
The data source is a separate module (`simulator.py`), so a real sensor stream could replace it later.

## Results (simulated data)

**AI vs fixed 50 ppm threshold (single test seed)**

| Method | F1 | FPR | Detection time (s) | Glitch alarms (of 9) |
|---|---|---|---|---|
| Fixed threshold 50 ppm | 0.821 | 0.0040 | 44.2 | 9 |
| Isolation Forest | 0.947 | 0.0160 | 8.6 | 0 |

**vs tuned thresholds (10 unseen seeds, mean)**

| Method | F1 | Detection time (s) | Glitch alarms (of 9) |
|---|---|---|---|
| Threshold 50 ppm | 0.789 | 50.7 | 9 |
| Threshold 10 ppm + 3 s persistence | 0.972 | 8.0 | 0 |
| Isolation Forest | 0.946 | 9.9 | 0 |

### Honest findings
- The model is much better than an **untuned** 50 ppm threshold.
- A **well-tuned low threshold with persistence matched or slightly beat** the model on this clean simulated data.
- Ablation: the gain comes mainly from **gas rate-of-change and rolling-median** features. Pressure and temperature features gave no measurable benefit in this simulation.

## Project structure
```
config.py          thresholds (fixed threshold, alarm level)
simulator.py       synthetic sensor data (batch + live)
features.py        feature engineering
train.py           trains Isolation Forest, saves models/iforest.joblib
risk.py            model score -> 0-100 risk and level
evaluate.py        baseline vs AI on one seed
compare_models.py  Isolation Forest vs One-Class SVM vs LOF
ablation.py        feature ablation study
final_eval.py      10-seed comparison with tuned thresholds
app.py             Streamlit dashboard
```

## Getting started

```bash
git clone https://github.com/[your-username]/cng-leak-early-warning.git
cd cng-leak-early-warning

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
python train.py                # creates models/iforest.joblib
streamlit run app.py
```

### Run the evaluations
```bash
python evaluate.py
python compare_models.py
python ablation.py
python final_eval.py
```

### Demo flow
1. Turn on **Start simulation**.
2. Press **Sensor glitch**: the fixed threshold alarms, the AI does not.
3. Turn on **Inject leak**: the AI raises the alarm well before gas reaches 50 ppm.
4. Press **Switch to PETROL** or **Vehicle stopped** to clear the alarm.

## Limitations
- All data is synthetic. Leak shape, noise and variable relationships are our assumptions.
- Gas values are scaled simulation units, not calibrated ppm of a real methane sensor.
- No sensor drift, calibration error or sensor failure is modelled.
- Simulated leaks stop abruptly, which causes a few false-alarm seconds right after a leak ends.
- The fuel-switch and "vehicle stopped" buttons are simulated.

## Future work
- Add sensor drift, higher noise and gas-sensor failure scenarios
- Validate on real sensors (for example MQ-4, BMP280 via ESP32) in a safe low-pressure setup
- Try sequence models such as an LSTM autoencoder
- Vehicle interface only after proper safety validation

## Tech stack
Python, NumPy, Pandas, scikit-learn, Altair, Streamlit

## Author
- Rohit Shahu Jambhulkar
