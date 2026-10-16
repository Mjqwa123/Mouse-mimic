"""Sample human-like mouse paths from a trained model.

    from generate import Mimic
    path = Mimic().generate_path((100, 100), (800, 450), radius=20)   # (N, 3) array of t, x, y
"""
import math

import numpy as np
import torch

from data import DT, MODEL_PATH, features, from_canonical, raw_condition
from model import MimicNet, sample_step


class Mimic:
    def __init__(self, path=MODEL_PATH):
        ck = torch.load(path, map_location="cpu")
        self.model = MimicNet(**ck["config"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.delta_scale = ck["delta_scale"]
        self.cond_mean = np.array(ck["cond_mean"])
        self.cond_std = np.array(ck["cond_std"])
        self.max_steps = ck["max_steps"]
        self.fitts = ck.get("fitts")
        self.val_ids = ck.get("val_ids", [])
        self.data_path = ck.get("data_path")

    def generate_path(self, start, target, radius=20.0, temperature=1.0, seed=None):
        """Return an (N, 3) array of (t seconds, x, y) from start to somewhere inside the target."""
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        distance = float(np.hypot(*(target - start)))
        if distance < 1.0:
            return np.array([[0.0, *start]])
        cond_n = (raw_condition(distance, radius) - self.cond_mean) / self.cond_std
        rng = np.random.default_rng(seed)
        r_c = max(radius, 1.0) / distance
        cap = int(self.max_steps * 1.5)
        if self.fitts:
            a, b, std = self.fitts
            cap = min(cap, int((a + b * math.log2(1.0 / r_c + 1.0) + 2.0 * std) / DT))

        pos = np.zeros(2)
        prev = np.zeros(2)
        h = None
        path = [pos.copy()]
        with torch.no_grad():
            for step in range(max(cap, 10)):
                x = torch.from_numpy(features(pos[0], pos[1], prev[0], prev[1], step, cond_n, r_c)).view(1, 1, -1)
                params, h = self.model(x, h)
                d, p_end = sample_step(params, temperature, rng)
                pos = pos + d * self.delta_scale
                prev = d
                path.append(pos.copy())
                # the still start looks like the still dwell before a click, so only
                # let the model end the move once it is at least halfway there
                if np.hypot(pos[0] - 1.0, pos[1]) < 0.5 and rng.random() < p_end:
                    break

        c = correct_miss(np.array(path), r_c, rng)
        c = land_on_target(c, r_c)
        xy = from_canonical(c, start, target)
        return np.column_stack([np.arange(len(xy)) * DT, xy])


def correct_miss(c, r_c, rng):
    """If the sampled movement stopped outside the target, add a brief pause and
    a quick minimum-jerk corrective submovement into it, as people do after a miss."""
    goal = np.array([1.0, 0.0])
    err = goal - c[-1]
    dist = float(np.hypot(*err))
    if dist <= r_c:
        return c
    aim = goal + rng.normal(0.0, 0.25 * r_c, 2)
    vec = aim - c[-1]
    length = float(np.hypot(*vec))
    pause = int(rng.uniform(0.03, 0.09) / DT)
    duration = 0.08 + 0.07 * math.log2(length / r_c + 1.0)  # Fitts' law for the correction
    n = max(int(duration / DT), 3)
    tau = np.arange(1, n + 1) / n
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    perp = np.array([-vec[1], vec[0]])
    bow = rng.normal(0.0, 0.08)
    sub = c[-1] + s[:, None] * vec + (bow * np.sin(np.pi * s))[:, None] * perp
    still = np.repeat(c[-1:], pause, axis=0)
    return np.vstack([c, still, sub])


def land_on_target(c, r_c, inside=0.8):
    """If the sampled path ends outside the target, bend its tail so it ends just
    inside. Paths that already land are left untouched."""
    goal = np.array([1.0, 0.0])
    err = goal - c[-1]
    dist = float(np.hypot(*err))
    if dist <= inside * r_c or len(c) < 3:
        return c
    shift = err * (1.0 - inside * r_c / dist)
    # bigger misses get corrected over a longer stretch so the bend stays smooth
    w = int(np.clip(len(c) * min(1.0, dist * 4.0), min(8, len(c) - 1), len(c) - 1))
    s = np.linspace(0.0, 1.0, w)
    ramp = s * s * (3.0 - 2.0 * s)
    c = c.copy()
    c[-w:] += ramp[:, None] * shift
    return c


if __name__ == "__main__":
    m = Mimic()
    p = m.generate_path((200, 200), (900, 500), radius=20, seed=1)
    print(f"{len(p)} points over {p[-1, 0]:.3f}s, ends at ({p[-1, 1]:.1f}, {p[-1, 2]:.1f})")
