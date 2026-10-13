# simulator.py : sensor ki jagah nakli (simulated) data banata hai

import numpy as np
import pandas as pd


def _base(n, rng):
    """Normal operation: gaadi chal rahi hai, koi leak nahi."""
    t = np.arange(n)
    speed = np.clip(40 + 20 * np.sin(t / 80) + rng.normal(0, 3, n), 0, 100)
    rpm = 800 + speed * 30 + rng.normal(0, 50, n)
    temp = 35 + 0.05 * speed + rng.normal(0, 0.3, n)
    pressure = 200 + 0.8 * (temp - 35) + rng.normal(0, 0.5, n)
    gas = np.clip(5 + rng.normal(0, 0.5, n), 0, None)
    return pd.DataFrame(dict(gas=gas, pressure=pressure, temp=temp, rpm=rpm, speed=speed))


def generate_stream(n=3000, leaks=(), spikes=(), seed=0):
    """
    n      = kitni readings (1 reading = 1 second)
    leaks  = [(start_index, duration), ...]  -> asli gradual leak (label = 1)
    spikes = [index, ...]                    -> sensor glitch, ek reading ka spike (label = 0)
    seed   = same seed = same data (result repeat ho sakta hai)
    """
    rng = np.random.default_rng(seed)
    df = _base(n, rng)
    df["label"] = 0

    for start, dur in leaks:
        k = np.arange(min(dur, n - start))
        idx = start + k
        rate = rng.uniform(0.5, 1.5)           # gas kitni tezi se badh rahi hai (ppm/sec)
        df.loc[idx, "gas"] += rate * k         # gas dheere dheere badhti hai
        df.loc[idx, "pressure"] -= 0.15 * k    # pressure girta hai
        df.loc[idx, "temp"] -= 0.02 * k        # gas expand hoke thanda hoti hai
        df.loc[idx, "label"] = 1

    for i in spikes:
        df.loc[i, "gas"] += 60                 # glitch: label 0 hi rahega

    return df


class LiveSimulator:
    """Dashboard ke liye: step() har baar ek naya reading deta hai."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.leak_t = 0
        self.rate = 1.0

    def step(self, leak=False, spike=False):
        r = self.rng
        self.t += 1
        speed = float(np.clip(40 + 20 * np.sin(self.t / 80) + r.normal(0, 3), 0, 100))
        rpm = 800 + speed * 30 + r.normal(0, 50)
        temp = 35 + 0.05 * speed + r.normal(0, 0.3)
        pressure = 200 + 0.8 * (temp - 35) + r.normal(0, 0.5)
        gas = max(0.0, 5 + r.normal(0, 0.5))

        if leak:
            if self.leak_t == 0:
                self.rate = r.uniform(0.5, 1.5)
            self.leak_t += 1
            gas += self.rate * self.leak_t
            pressure -= 0.15 * self.leak_t
            temp -= 0.02 * self.leak_t
        else:
            self.leak_t = 0

        if spike:
            gas += 60

        return dict(gas=gas, pressure=pressure, temp=temp, rpm=rpm, speed=speed)