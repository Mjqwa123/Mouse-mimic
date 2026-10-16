"""Train the movement model on recorded moves.

    python train.py
    python train.py --epochs 100 --data data/moves.jsonl
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from data import DATA_PATH, DT, MODEL_PATH, load_moves, move_to_canonical, training_example
from model import MimicNet, loss_fn


def make_batch(examples):
    xs, ys, ends = zip(*examples)
    lengths = torch.tensor([len(x) for x in xs])
    x = pad_sequence([torch.from_numpy(a) for a in xs], batch_first=True)
    y = pad_sequence([torch.from_numpy(a) for a in ys], batch_first=True)
    e = pad_sequence([torch.from_numpy(a) for a in ends], batch_first=True)
    mask = (torch.arange(x.shape[1])[None, :] < lengths[:, None]).float()
    return x, y, e, mask


def evaluate(model, examples, batch_size):
    model.eval()
    total, steps = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            x, y, e, mask = make_batch(examples[i:i + batch_size])
            nll, bce = loss_fn(model(x)[0], y, e, mask)
            n = mask.sum().item()
            total += (nll.item() + bce.item()) * n
            steps += n
    return total / max(steps, 1)


def train(data=DATA_PATH, out=MODEL_PATH, epochs=60, batch_size=32, lr=1e-3, hidden=128, mixtures=8,
          mirror=False, seed=0, log=print, should_stop=lambda: False):
    """Train on the recordings in `data` and save the best model to `out`. Returns the best val loss."""
    data, out = Path(data), Path(out)
    if not data.exists():
        raise ValueError(f"No recordings at {data}. Record some moves first.")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    moves = load_moves(data)
    ids, paths, conds = [], [], []
    for i, m in enumerate(moves):
        if len(m["points"]) < 3:
            continue
        c, cond = move_to_canonical(m, rng)
        if len(c) < 3:
            continue
        ids.append(i)
        paths.append(c)
        conds.append(cond)
    n = len(paths)
    if n < 10:
        raise ValueError(f"Only {n} usable moves; record at least a few dozen (300+ recommended).")
    if n < 100:
        log(f"Warning: only {n} moves. It will train, but 300+ gives much more human-looking output.")

    perm = rng.permutation(n)
    n_val = max(1, n // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    deltas = np.concatenate([np.diff(paths[i], axis=0) for i in train_idx])
    delta_scale = float(deltas.std())
    cond_arr = np.array([conds[i] for i in train_idx])
    cond_mean = cond_arr.mean(0)
    cond_std = np.where(cond_arr.std(0) > 1e-6, cond_arr.std(0), 1.0)

    def build(indices, mirror):
        out = []
        for i in indices:
            cn = (conds[i] - cond_mean) / cond_std
            out.append(training_example(paths[i], conds[i], cn, delta_scale))
            if mirror:
                out.append(training_example(paths[i] * [1.0, -1.0], conds[i], cn, delta_scale))
        return out

    train_ex = build(train_idx, mirror)
    val_ex = build(val_idx, False)
    max_steps = max(len(p) for p in paths)
    # Fitts' law fit of duration: caps generation so a stalled sample can't crawl forever
    index = np.array([math.log2(1.0 / math.exp(conds[i][1]) + 1.0) for i in train_idx])
    duration = np.array([(len(paths[i]) - 1) * DT for i in train_idx])
    fitts_b, fitts_a = np.polyfit(index, duration, 1)
    fitts_std = float((duration - (fitts_a + fitts_b * index)).std())
    log(f"{n} moves ({len(train_idx)} train, {n_val} val), longest {max_steps * DT:.2f}s, "
          f"{sum(len(e[0]) for e in train_ex)} training steps")

    model = MimicNet(hidden=hidden, mixtures=mixtures)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = math.inf

    for epoch in range(1, epochs + 1):
        if should_stop():
            log("Stopped early.")
            break
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_ex))
        total, steps = 0.0, 0.0
        for b in range(0, len(order), batch_size):
            x, y, e, mask = make_batch([train_ex[j] for j in order[b:b + batch_size]])
            nll, bce = loss_fn(model(x)[0], y, e, mask)
            loss = nll + bce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            k = mask.sum().item()
            total += loss.item() * k
            steps += k
        sched.step()
        val = evaluate(model, val_ex, batch_size)
        mark = ""
        if val < best:
            best = val
            mark = "  *saved"
            torch.save({
                "state": model.state_dict(),
                "config": model.config,
                "delta_scale": delta_scale,
                "cond_mean": [float(v) for v in cond_mean],
                "cond_std": [float(v) for v in cond_std],
                "max_steps": int(max_steps),
                "fitts": [float(fitts_a), float(fitts_b), fitts_std],
                "dt": DT,
                "val_ids": [int(ids[i]) for i in val_idx],
                "data_path": str(data.resolve()),
            }, out)
        log(f"epoch {epoch:3d}  train {total / steps:8.4f}  val {val:8.4f}  ({time.perf_counter() - t0:.1f}s){mark}")

    log(f"Best val loss {best:.4f}. Model saved to {out}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mixtures", type=int, default=8)
    ap.add_argument("--mirror", action="store_true", help="also train on vertically mirrored copies of each move")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        train(args.data, args.out, args.epochs, args.batch_size, args.lr, args.hidden, args.mixtures,
              args.mirror, args.seed)
    except ValueError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
