# app.py : Streamlit dashboard v7 (instrument cluster: gauges, warning lights, odometer)

import io
import time
import wave

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from simulator import LiveSimulator
from risk import load, risk_from_df, level
from config import GAS_THRESHOLD, ALARM_RISK

st.set_page_config(page_title="CNG Early Warning", page_icon="🔥", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;}
.status{padding:18px 24px;border-radius:14px;color:white;font-size:1.7rem;font-weight:700;margin-bottom:14px}
.status small{display:block;font-size:0.95rem;font-weight:400;opacity:.9}
.card{background:#1c2230;border-radius:12px;padding:14px 18px;border:1px solid #2b3345;height:100%}
.card .l{color:#9aa4b8;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:1.9rem;font-weight:700;line-height:1.3}
.card .s{color:#9aa4b8;font-size:.8rem}
.gbar{height:18px;border-radius:9px;position:relative;margin:14px 0 4px;
 background:linear-gradient(90deg,#2ecc71 0%,#3498db 25%,#f39c12 50%,#e74c3c 75%,#c0392b 100%)}
.gbar .m{position:absolute;top:-5px;width:4px;height:28px;background:white;border-radius:2px;box-shadow:0 0 6px #000}
.scale{display:flex;justify-content:space-between;color:#9aa4b8;font-size:.75rem;margin-bottom:12px}
@keyframes blink{0%,100%{background:#b30000}50%{background:#ff2b2b}}
.alert{padding:20px 24px;border-radius:14px;color:white;font-size:1.6rem;font-weight:800;
 margin-bottom:14px;text-align:center;animation:blink 0.7s infinite}
.alert small{display:block;font-size:1rem;font-weight:500}
.ctitle{font-weight:700;font-size:1.05rem;margin:18px 0 2px 0}
.ctitle span{font-weight:400;font-size:.8rem;color:#9aa4b8}

/* audio player screen ke bahar (awaaz chalti rahegi) */
div[data-testid="stAudio"], div.element-container:has(audio){
 position:fixed !important;left:-9999px !important;bottom:0;width:1px !important;height:1px !important;overflow:hidden}

/* ---- instrument cluster ---- */
.cluster{background:#151a26;border:1px solid #2b3345;border-radius:16px;padding:10px 6px 4px 6px;margin-bottom:14px}
.g{position:relative;width:220px;height:170px;margin:0 auto;overflow:hidden}
.arc{position:absolute;left:0;top:0;width:220px;height:220px;border-radius:50%}
.hole{position:absolute;left:30px;top:30px;width:160px;height:160px;border-radius:50%;background:#151a26}
.needle{position:absolute;left:108px;top:15px;width:4px;height:95px;background:#fff;border-radius:2px;
 transform-origin:50% 100%;box-shadow:0 0 6px #000}
.hub{position:absolute;left:100px;top:102px;width:16px;height:16px;border-radius:50%;background:#fff}
.gv{position:absolute;left:0;width:100%;top:120px;text-align:center;font-size:1.5rem;font-weight:700;color:#fff}
.gv small{font-size:.75rem;color:#9aa4b8;margin-left:3px}
.gl{position:absolute;left:0;width:100%;top:150px;text-align:center;font-size:.72rem;color:#9aa4b8;
 text-transform:uppercase;letter-spacing:.08em}
.lamps{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin:6px 0 12px 0}
.lamp{display:flex;flex-direction:column;align-items:center;width:92px;font-size:.62rem;color:#6f7a90;
 text-transform:uppercase;letter-spacing:.05em;text-align:center}
.lamp .c{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;
 font-size:1.4rem;background:#222a3b;border:2px solid #2f394d;filter:grayscale(1);opacity:.5;margin-bottom:4px}
.lamp.red .c{background:#c0171d;border-color:#ff5a5f;filter:none;opacity:1;box-shadow:0 0 16px #ff2b2b;animation:pulse .7s infinite}
.lamp.amber .c{background:#b87400;border-color:#ffc247;filter:none;opacity:1;box-shadow:0 0 14px #ffb020}
.lamp.green .c{background:#12724a;border-color:#4be3a0;filter:none;opacity:1;box-shadow:0 0 12px #2ecc71}
.lamp.red,.lamp.amber,.lamp.green{color:#fff}
@keyframes pulse{0%,100%{box-shadow:0 0 6px #ff2b2b}50%{box-shadow:0 0 24px #ff2b2b}}
.odo{display:flex;justify-content:center;align-items:center;gap:4px;margin:2px 0 10px 0}
.odo .d{background:#0b0e15;border:1px solid #2f394d;border-radius:5px;color:#e8ecf5;
 font-family:monospace;font-size:1.5rem;width:30px;text-align:center;padding:2px 0}
.odo .u{color:#9aa4b8;font-size:.8rem;margin-left:8px}
</style>
""", unsafe_allow_html=True)

st.title("🔥 AI-Based CNG Leakage Early-Warning System")
st.caption("Simulated sensor data  |  Isolation Forest anomaly detection  |  Research prototype")

bundle = st.cache_resource(load)()

LEVEL_COLOR = {"SAFE": "#1e8e5a", "WATCH": "#2a6fdb", "WARNING": "#e08a00", "DANGER": "#d62728"}
LEVEL_MSG = {
    "SAFE": "Normal operation",
    "WATCH": "Slight abnormal behaviour",
    "WARNING": "Possible gas leakage detected",
    "DANGER": "Stop the vehicle and check immediately",
}

# ---------------- sounds ----------------
SR = 16000


def tone(freq, dur, kind="sine", amp=0.7):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    s = np.sin(2 * np.pi * freq * t)
    if kind == "square":
        s = np.sign(s) * 0.6
    ramp = int(SR * 0.005)
    env = np.ones_like(s)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return s * env * amp


def gap(dur):
    return np.zeros(int(SR * dur))


def to_wav(x):
    x = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.tobytes())
    return buf.getvalue()


@st.cache_resource
def build_sounds():
    return {
        "Fast beeps (car warning)": to_wav(np.concatenate(
            [tone(1000, 0.12), gap(0.12)] * 4 + [gap(0.4)])),
        "Buzzer": to_wav(np.concatenate(
            [tone(500, 0.25, "square"), gap(0.2)] * 2)),
        "Klaxon": to_wav(np.concatenate(
            [tone(440, 0.4, "square"), tone(330, 0.4, "square")])),
        "Siren (ambulance)": to_wav(np.concatenate(
            [tone(900, 0.25), tone(1300, 0.25)])),
    }


SOUNDS = build_sounds()

# ---------------- UI helpers ----------------
GRAD_UP = "conic-gradient(from -90deg,#2ecc71 0deg,#3498db 50deg,#f39c12 100deg,#e74c3c 150deg,#c0392b 180deg,#1c2230 180deg 360deg)"
GRAD_DOWN = "conic-gradient(from -90deg,#c0392b 0deg,#e74c3c 40deg,#f39c12 80deg,#2ecc71 130deg,#2ecc71 180deg,#1c2230 180deg 360deg)"
GRAD_NEUTRAL = "conic-gradient(from -90deg,#2a6fdb 0deg,#5dade2 180deg,#1c2230 180deg 360deg)"


def card(label, value, sub="", color="#ffffff"):
    return (f"<div class='card'><div class='l'>{label}</div>"
            f"<div class='v' style='color:{color}'>{value}</div>"
            f"<div class='s'>{sub}</div></div>")


def gauge(label, value, lo, hi, unit, grad, decimals=0):
    frac = min(max((value - lo) / (hi - lo), 0.0), 1.0)
    ang = -90 + frac * 180
    txt = f"{value:.{decimals}f}"
    return (f"<div class='g'><div class='arc' style='background:{grad}'></div><div class='hole'></div>"
            f"<div class='needle' style='transform:rotate({ang:.1f}deg)'></div><div class='hub'></div>"
            f"<div class='gv'>{txt}<small>{unit}</small></div><div class='gl'>{label}</div></div>")


def lamp(icon, text, state):
    return f"<div class='lamp {state}'><div class='c'>{icon}</div>{text}</div>"





def odometer_html(km):
    s = f"{km:07.2f}"
    parts = ""
    for ch in s:
        parts += f"<div class='d'>{ch}</div>" if ch != "." else "<div class='d' style='width:12px'>.</div>"
    return f"<div class='odo'>{parts}<span class='u'>km (simulated)</span></div>"


def fmt(x):
    return "not yet" if x is None else f"{x} s after leak start"


def ctitle(col, text, sub=""):
    col.markdown(f"<div class='ctitle'>{text} <span>{sub}</span></div>", unsafe_allow_html=True)


def line_chart(data, col, ytitle, color, y_range, rule=None):
    d = data.tail(120)
    main = alt.Chart(d).mark_line(color=color, strokeWidth=2.5).encode(
        x=alt.X("t:Q", title="Time (s)", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{col}:Q", title=ytitle, scale=alt.Scale(domain=list(y_range))),
    )
    if rule is not None:
        line = alt.Chart(pd.DataFrame({"y": [rule]})).mark_rule(
            color="red", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
        main = main + line
    return main.properties(height=220)


# ---------------- state ----------------
def new_state():
    st.session_state.sim = LiveSimulator(seed=1)
    st.session_state.hist = []
    st.session_state.risks = []
    st.session_state.base_count = 0
    st.session_state.ai_count = 0
    st.session_state.leak_start = None
    st.session_state.base_first = None
    st.session_state.ai_first = None
    st.session_state.latched = False
    st.session_state.mode = "CNG"
    st.session_state.leak_on = False
    st.session_state.odo = 0.0


def do_reset():
    new_state()


def go_petrol():
    st.session_state.latched = False
    st.session_state.mode = "PETROL"
    st.session_state.leak_on = False


def vehicle_stopped():
    st.session_state.latched = False
    st.session_state.mode = "STOPPED"


if "odo" not in st.session_state:
    new_state()

# ---------------- sidebar ----------------
st.sidebar.header("Controls")
run = st.sidebar.toggle("▶ Start simulation")
leak = st.sidebar.toggle("💨 Inject leak", key="leak_on")
spike = st.sidebar.button("⚡ Sensor glitch (1 spike)")
st.sidebar.button("↺ Reset", on_click=do_reset)

st.sidebar.divider()
st.sidebar.subheader("Driver actions")
st.sidebar.button("⛽ Switch to PETROL", on_click=go_petrol,
                  disabled=not st.session_state.latched, use_container_width=True)
st.sidebar.button("🛑 Vehicle stopped", on_click=vehicle_stopped,
                  disabled=not st.session_state.latched, use_container_width=True)
st.sidebar.caption("Alarm tab tak bajega jab tak driver inme se ek action na le.")

st.sidebar.divider()
sound_choice = st.sidebar.selectbox("🔊 Alarm sound", list(SOUNDS.keys()), index=0)

st.sidebar.divider()
st.sidebar.subheader("Risk levels")
st.sidebar.markdown(
    "- 🟢 **SAFE**: 0 to 25\n"
    "- 🔵 **WATCH**: 25 to 50\n"
    "- 🟠 **WARNING**: 50 to 75 (alarm)\n"
    "- 🔴 **DANGER**: 75 to 100"
)
st.sidebar.caption("Red dashed line = alarm limit. Gas chart: fixed threshold (50 ppm). "
                   "Risk chart: AI alarm (50).")

# ---------------- page layout (siren slot sabse neeche) ----------------
box = st.empty()
siren_slot = st.empty()
if st.session_state.latched:
    siren_slot.audio(SOUNDS[sound_choice], format="audio/wav", loop=True, autoplay=True)
else:
    siren_slot.empty()

if not run:
    st.info("Sidebar : ▶ Start simulation")

first = True
while run:
    sim = st.session_state.sim
    spike_now = bool(spike and first)
    reading = sim.step(leak=leak, spike=spike_now)
    reading["t"] = sim.t
    first = False

    st.session_state.hist.append(reading)
    st.session_state.hist = st.session_state.hist[-300:]
    df = pd.DataFrame(st.session_state.hist)

    risk = float(risk_from_df(df, bundle).iloc[-1])
    st.session_state.risks.append({"t": sim.t, "risk": risk})
    st.session_state.risks = st.session_state.risks[-300:]
    rdf = pd.DataFrame(st.session_state.risks)

    lvl = level(risk)
    base_alarm = reading["gas"] > GAS_THRESHOLD
    ai_alarm = risk >= ALARM_RISK
    st.session_state.base_count += int(base_alarm)
    st.session_state.ai_count += int(ai_alarm)

    # ---- detection time tracking ----
    if leak and sim.leak_t == 1:
        st.session_state.leak_start = sim.t
        st.session_state.base_first = None
        st.session_state.ai_first = None
    ls = st.session_state.leak_start
    if leak and ls is not None:
        if base_alarm and not spike_now and st.session_state.base_first is None:
            st.session_state.base_first = sim.t - ls
        if ai_alarm and st.session_state.ai_first is None:
            st.session_state.ai_first = sim.t - ls

    # ---- ALARM LATCH ----
    if ai_alarm and st.session_state.mode == "CNG" and not st.session_state.latched:
        st.session_state.latched = True
        st.rerun()

    mode = st.session_state.mode
    shown_speed = 0.0 if mode == "STOPPED" else reading["speed"]
    st.session_state.odo += shown_speed / 3600.0

    gas_max = max(70, float(df["gas"].max()) + 10)
    p_lo = min(180.0, float(df["pressure"].min()) - 3)
    p_hi = max(210.0, float(df["pressure"].max()) + 3)
    t_lo = min(30.0, float(df["temp"].min()) - 1)
    t_hi = max(40.0, float(df["temp"].max()) + 1)

    with box.container():
        if st.session_state.latched:
            st.markdown(
                "<div class='alert'>🚨 GAS LEAK ALERT 🚨"
                "<small>Switch to PETROL or STOP the vehicle (sidebar: Driver actions)</small></div>",
                unsafe_allow_html=True)
        elif mode == "PETROL":
            st.success("⛽ Driver switched to PETROL. CNG line isolated, alarm cleared.")
        elif mode == "STOPPED":
            st.success("🛑 Vehicle stopped by driver. Alarm cleared.")

        st.markdown(
            f"<div class='status' style='background:{LEVEL_COLOR[lvl]}'>{lvl}"
            f"<small>{LEVEL_MSG[lvl]}  |  Fuel mode: {mode}</small></div>", unsafe_allow_html=True)

        # ---- warning lights ----
        lamps = (
            lamp("🚨", "Leak alert", "red" if st.session_state.latched else "off")
            + lamp("🔥", "Gas high", "amber" if reading["gas"] > 15 else "off")
            + lamp("⬇️", "Low pressure", "amber" if reading["pressure"] < 195 else "off")
            + lamp("📡", "Sensor?", "amber" if (base_alarm and not ai_alarm) else "off")
            + lamp("⛽", "CNG", "green" if mode == "CNG" else "off")
            + lamp("🛢️", "Petrol", "green" if mode == "PETROL" else "off")
            + lamp("🛑", "Stopped", "amber" if mode == "STOPPED" else "off")
        )

        # ---- instrument cluster ----
        st.markdown(f"<div class='lamps'>{lamps}</div>", unsafe_allow_html=True)
        g = st.columns(4)
        g[0].markdown(f"<div class='cluster'>{gauge('Speed', shown_speed, 0, 120, 'km/h', GRAD_NEUTRAL)}</div>",
                      unsafe_allow_html=True)
        g[1].markdown(f"<div class='cluster'>{gauge('Gas (simulated ppm)', reading['gas'], 0, 100, 'ppm', GRAD_UP)}</div>",
                      unsafe_allow_html=True)
        g[2].markdown(f"<div class='cluster'>{gauge('Pressure', reading['pressure'], 170, 210, 'bar', GRAD_DOWN)}</div>",
                      unsafe_allow_html=True)
        g[3].markdown(f"<div class='cluster'>{gauge('Temperature', reading['temp'], 20, 50, '°C', GRAD_UP, 1)}</div>",
                      unsafe_allow_html=True)
        st.markdown(odometer_html(st.session_state.odo), unsafe_allow_html=True)

        # ---- AI risk bar ----
        st.markdown(
            f"<div class='gbar'><div class='m' style='left:calc({min(risk, 100):.0f}% - 2px)'></div></div>"
            "<div class='scale'><span>0 SAFE</span><span>25 WATCH</span>"
            "<span>50 WARNING</span><span>75 DANGER</span><span>100</span></div>",
            unsafe_allow_html=True)
        st.caption(f"AI risk score: {risk:.0f} / 100 (alarm at {ALARM_RISK})")

        p1, p2 = st.columns(2)
        p1.markdown(card(
            f"Fixed threshold ({GAS_THRESHOLD} ppm)",
            "ALARM" if base_alarm else "OK",
            f"Alarm seconds: {st.session_state.base_count}  |  First detection: {fmt(st.session_state.base_first)}",
            "#e74c3c" if base_alarm else "#58d68d"), unsafe_allow_html=True)
        p2.markdown(card(
            "AI model (Isolation Forest)",
            "ALARM" if ai_alarm else "OK",
            f"Alarm seconds: {st.session_state.ai_count}  |  First detection: {fmt(st.session_state.ai_first)}",
            "#e74c3c" if ai_alarm else "#58d68d"), unsafe_allow_html=True)

        st.write("")
        r1a, r1b = st.columns(2)
        ctitle(r1a, "Gas concentration", "ppm = parts per million (simulated units)")
        r1a.altair_chart(line_chart(df, "gas", "ppm", "#5dade2", (0, gas_max),
                                    rule=GAS_THRESHOLD), use_container_width=True)
        ctitle(r1b, "AI risk score", "0 to 100")
        r1b.altair_chart(line_chart(rdf, "risk", "risk", "#ec7063", (0, 100),
                                    rule=ALARM_RISK), use_container_width=True)
        r2a, r2b = st.columns(2)
        ctitle(r2a, "Pressure", "bar")
        r2a.altair_chart(line_chart(df, "pressure", "bar", "#58d68d", (p_lo, p_hi)),
                         use_container_width=True)
        ctitle(r2b, "Temperature", "°C")
        r2b.altair_chart(line_chart(df, "temp", "°C", "#f5b041", (t_lo, t_hi)),
                         use_container_width=True)

    time.sleep(1)