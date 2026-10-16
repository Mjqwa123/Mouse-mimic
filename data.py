"""Shared data handling: loading recorded moves, resampling, canonical frame and model features."""
import ctypes
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

DT = 0.008  # resample step in seconds (125 Hz)
if getattr(sys, "frozen", False):
    # packaged .exe: keep recordings and the model in a fixed per-user folder
    ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MouseMimic"
else:
    ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "moves.jsonl"
MODEL_PATH = ROOT / "models" / "mimic.pt"
N_FEATURES = 10


def make_dpi_aware():
    """Make tkinter and the Win32 cursor calls agree on physical pixel coordinates."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def load_moves(path=DATA_PATH):
    """Return the recorded moves as a list of dicts, in file order."""
    moves = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                moves.append(json.loads(line))
    return moves


def resample(points, start, move_thresh=2.0):
    """Trim the still reaction time before the hand starts moving, then resample
    to a fixed DT grid. Returns (N, 2) screen-space positions."""
    p = np.asarray(points, dtype=float)
    t, xy = p[:, 0], p[:, 1:3]
    moved = np.linalg.norm(xy - np.asarray(start, dtype=float), axis=1) > move_thresh
    if moved.any():
        i = max(int(np.argmax(moved)) - 1, 0)
        t, xy = t[i:], xy[i:]
    t = t - t[0]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, xy = t[keep], xy[keep]
    n = max(int(math.ceil(t[-1] / DT)) + 1, 2)
    grid = np.minimum(np.arange(n) * DT, t[-1])
    return np.column_stack([np.interp(grid, t, xy[:, 0]), np.interp(grid, t, xy[:, 1])])


def _frame(start, target):
    s = np.asarray(start, dtype=float)
    v = np.asarray(target, dtype=float) - s
    d = float(np.hypot(*v))
    c, sn = v / d
    rot = np.array([[c, sn], [-sn, c]])  # rotates the start->target vector onto +x
    return s, rot, d


def to_canonical(xy, start, target):
    """Screen space -> frame where start is (0,0) and target is (1,0)."""
    s, rot, d = _frame(start, target)
    return (np.asarray(xy, dtype=float) - s) @ rot.T / d


def from_canonical(cxy, start, target):
    s, rot, d = _frame(start, target)
    return (np.asarray(cxy, dtype=float) * d) @ rot + s


def raw_condition(distance, radius):
    """Per-move conditioning: movement distance and target difficulty (Fitts)."""
    return np.array([math.log(distance), math.log(max(radius, 1.0) / distance)])


def features(x, y, pdx, pdy, step, cond_n, r_c):
    """Per-step model input. Works on scalars or equal-length arrays.
    pdx/pdy are the previous step's delta in scaled units; cond_n is the
    normalized conditioning vector; r_c is the target radius in canonical units."""
    x, y, pdx, pdy, step = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, y, pdx, pdy, step)))
    shape = x.shape
    # distance to the target in target radii (log): below 0 means "inside", which drives the click decision
    in_target = np.log(np.hypot(1.0 - x, y) / r_c + 0.05)
    cols = [x, y, pdx, pdy, 1.0 - x, -y, step * DT, np.full(shape, cond_n[0]), np.full(shape, cond_n[1]), in_target]
    return np.stack(cols, axis=-1).astype(np.float32)


def move_to_canonical(move, rng=None):
    """Return (canonical positions (T, 2), raw condition) for one recorded move.
    With rng, adds +-0.5px dequantization noise so the model doesn't learn integer pixels."""
    xy = resample(move["points"], move["start"])
    if rng is not None:
        xy[1:] += rng.uniform(-0.5, 0.5, size=xy[1:].shape)
    distance = float(np.hypot(*(np.asarray(move["target"], float) - np.asarray(move["start"], float))))
    c = to_canonical(xy, move["start"], move["target"])
    c[0] = 0.0
    return c, raw_condition(distance, move["radius"])


def training_example(c, cond, cond_n, delta_scale):
    """Inputs, next-step targets and end-of-move flags for one canonical path."""
    d = np.diff(c, axis=0) / delta_scale
    prev = np.vstack([np.zeros((1, 2)), d[:-1]])
    x = features(c[:-1, 0], c[:-1, 1], prev[:, 0], prev[:, 1], np.arange(len(d)), cond_n, math.exp(cond[1]))
    end = np.zeros(len(d), dtype=np.float32)
    end[-1] = 1.0
    return x, d.astype(np.float32), end


def path_stats(path):
    """Duration (s), peak speed (px/s) and path length / straight-line ratio of a (N, 3) t,x,y path."""
    path = np.asarray(path, dtype=float)
    xy = path[:, 1:3]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dts = np.maximum(np.diff(path[:, 0]), 1e-6)
    straight = max(float(np.linalg.norm(xy[-1] - xy[0])), 1e-6)
    return {
        "duration": float(path[-1, 0] - path[0, 0]),
        "peak_speed": float((steps / dts).max()) if len(steps) else 0.0,
        "ratio": float(steps.sum() / straight),
    }
