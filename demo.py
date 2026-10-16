"""See what the trained model does.

Compare mode (default), full screen:
    Space   replay a real held-out move (gray) next to a generated one (orange)
    Click   generate a path from the last click to this one
    G       10 generated paths for the last start/target, to show the variation
    C       clear          Esc  quit

Cursor mode moves your real cursor (it never clicks):
    python demo.py --cursor                    # 5 moves to random points
    python demo.py --cursor --to 800 400       # one move to (800, 400)
    python demo.py --cursor --repeat 20 --temperature 0.8
Press Esc or touch the mouse to stop it.
"""
import argparse
import ctypes
import random
import sys
import time
import tkinter as tk

import numpy as np

from data import DATA_PATH, DT, MODEL_PATH, load_moves, make_dpi_aware, path_stats, resample
from generate import Mimic

REAL_COLOR, GEN_COLOR, MULTI_COLOR = "#8a8f99", "#ff9f43", "#4fb3ff"


class CompareDemo:
    def __init__(self, mimic, master=None):
        self.mimic = mimic
        self.real_moves = []
        for path in (mimic.data_path, DATA_PATH):
            try:
                moves = load_moves(path)
            except (OSError, TypeError):
                continue
            ids = [i for i in mimic.val_ids if i < len(moves)] or range(len(moves))
            self.real_moves = [moves[i] for i in ids]
            break

        self.master = master
        self.root = tk.Toplevel(master) if master else tk.Tk()
        self.root.title("Mouse Mimic - demo")
        self.root.attributes("-fullscreen", True)
        self.canvas = tk.Canvas(self.root, bg="#15171c", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.root.update()

        panel = tk.Frame(self.root, bg="#22252c")
        panel.place(x=20, rely=1.0, y=-20, anchor="sw")
        self.temp = tk.DoubleVar(value=1.0)
        self.radius = tk.DoubleVar(value=20)
        for label, var, lo, hi, res in (("temperature", self.temp, 0.1, 2.0, 0.05),
                                        ("target radius", self.radius, 3, 80, 1)):
            tk.Scale(panel, label=label, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal",
                     length=220, bg="#22252c", fg="#c8ccd4", highlightthickness=0, troughcolor="#3a3f4a").pack(
                side="left", padx=8, pady=6)

        self.info = self.canvas.create_text(
            20, 20, anchor="nw", fill="#c8ccd4", font=("Consolas", 12),
            text="Space: real vs generated   Click: generate to here   G: 10 samples   C: clear   Esc: quit")
        self.stats = self.canvas.create_text(20, 50, anchor="nw", fill="#c8ccd4", font=("Consolas", 12), text="")
        self.last = None
        self.last_target = None
        self.anims = []

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<space>", lambda e: self.real_vs_generated())
        self.root.bind("g", lambda e: self.many())
        self.root.bind("c", lambda e: self.clear())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def to_canvas(self, x, y):
        return x - self.canvas.winfo_rootx(), y - self.canvas.winfo_rooty()

    def clear(self):
        self.anims = []
        self.canvas.delete("path")
        self.canvas.itemconfigure(self.stats, text="")

    def draw_target(self, target, radius):
        x, y = self.to_canvas(*target)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="#e0644f", width=2, tags="path")

    def generate(self, start, target, radius):
        return self.mimic.generate_path(start, target, radius=radius, temperature=self.temp.get())

    def on_click(self, e):
        p = (e.x_root, e.y_root)
        if self.last is None:
            self.clear()
            x, y = self.to_canvas(*p)
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=GEN_COLOR, outline="", tags="path")
        else:
            self.clear()
            r = self.radius.get()
            self.draw_target(p, r)
            path = self.generate(self.last, p, r)
            self.show_stats([("generated", path)])
            self.animate([(path, GEN_COLOR, 3)])
            self.last_target = (self.last, p, r)
        self.last = p

    def real_vs_generated(self):
        if not self.real_moves:
            self.canvas.itemconfigure(self.stats, text="No recorded moves found to compare against.")
            return
        self.clear()
        m = random.choice(self.real_moves)
        xy = resample(m["points"], m["start"])
        real = np.column_stack([np.arange(len(xy)) * DT, xy])
        gen = self.generate(m["start"], m["target"], m["radius"])
        self.draw_target(m["target"], m["radius"])
        self.show_stats([("real", real), ("generated", gen)])
        self.animate([(real, REAL_COLOR, 3), (gen, GEN_COLOR, 3)])
        self.last = tuple(m["points"][-1][1:3])
        self.last_target = (m["start"], m["target"], m["radius"])

    def many(self):
        if self.last_target is None:
            return
        start, target, r = self.last_target
        self.clear()
        self.draw_target(target, r)
        paths = [self.generate(start, target, r) for _ in range(10)]
        self.show_stats([(f"sample {i + 1}", p) for i, p in enumerate(paths)])
        self.animate([(p, MULTI_COLOR, 2) for p in paths])

    def show_stats(self, rows):
        lines = [f"{'':12} {'duration':>9} {'peak speed':>12} {'path/straight':>14}"]
        for name, p in rows:
            s = path_stats(p)
            lines.append(f"{name:12} {s['duration']:8.3f}s {s['peak_speed']:9.0f}px/s {s['ratio']:14.3f}")
        self.canvas.itemconfigure(self.stats, text="\n".join(lines))

    def animate(self, paths):
        """Draw each path progressively at its real speed."""
        token = object()
        self.anims = [token]
        drawn = [1] * len(paths)
        t0 = time.perf_counter()

        def tick():
            if self.anims != [token]:
                return
            elapsed = time.perf_counter() - t0
            busy = False
            for i, (p, color, width) in enumerate(paths):
                n = int(np.searchsorted(p[:, 0], elapsed, side="right"))
                if n > drawn[i]:
                    seg = p[drawn[i] - 1:n, 1:3]
                    coords = [c for x, y in seg for c in self.to_canvas(x, y)]
                    self.canvas.create_line(*coords, fill=color, width=width, capstyle="round", tags="path")
                    drawn[i] = n
                busy |= drawn[i] < len(p)
            if busy:
                self.root.after(8, tick)

        tick()

    def run(self):
        if self.master:
            self.root.focus_force()
            self.master.wait_window(self.root)
        else:
            self.root.mainloop()


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def move_cursor(mimic, to=None, repeat=None, radius=5.0, temperature=1.0, pause=0.6,
                log=print, should_stop=lambda: False):
    """Drive the real cursor along generated paths. Never clicks. Stops on Esc, if the
    user moves the mouse, or when should_stop() is true. Returns a status message."""
    if sys.platform != "win32":
        return "Cursor mode uses the Win32 API and only works on Windows."
    user32 = ctypes.windll.user32
    sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    def cursor():
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def esc_down():
        return bool(user32.GetAsyncKeyState(0x1B) & 0x8000)

    def stopped():
        return esc_down() or should_stop()

    for i in range(3, 0, -1):
        log(f"Moving the cursor in {i}... (Esc or touch the mouse to stop)")
        end = time.perf_counter() + 1
        while time.perf_counter() < end:
            if stopped():
                return "Stopped."
            time.sleep(0.01)

    repeat = 1 if to and repeat is None else (repeat or 5)
    for n in range(repeat):
        start = cursor()
        if to:
            target = tuple(to)
        else:
            m = 80
            while True:
                target = (random.uniform(m, sw - m), random.uniform(m, sh - m))
                if np.hypot(target[0] - start[0], target[1] - start[1]) >= 150:
                    break
        path = mimic.generate_path(start, target, radius=radius, temperature=temperature)
        last = start
        t0 = time.perf_counter()
        for t, x, y in path[1:]:
            while time.perf_counter() - t0 < t:
                if stopped():
                    return "Stopped."
                time.sleep(0.0005)
            cx, cy = cursor()
            if abs(cx - last[0]) > 3 or abs(cy - last[1]) > 3:
                return "Stopped (you moved the mouse)."
            last = (min(max(round(x), 0), sw - 1), min(max(round(y), 0), sh - 1))
            user32.SetCursorPos(*last)
        log(f"move {n + 1}/{repeat}: {start} -> ({target[0]:.0f}, {target[1]:.0f}) in {path[-1, 0]:.3f}s")
        pause_end = time.perf_counter() + pause
        while time.perf_counter() < pause_end:
            if stopped():
                return "Stopped."
            time.sleep(0.01)
    return "Done."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--cursor", action="store_true", help="move the real cursor instead of opening the window")
    ap.add_argument("--to", type=float, nargs=2, metavar=("X", "Y"), help="cursor mode: move to this point")
    ap.add_argument("--repeat", type=int, help="cursor mode: number of moves (default 5, or 1 with --to)")
    ap.add_argument("--radius", type=float, default=5, help="cursor mode: how close it must land, in px")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pause", type=float, default=0.6, help="cursor mode: seconds between moves")
    args = ap.parse_args()

    make_dpi_aware()
    try:
        mimic = Mimic(args.model)
    except FileNotFoundError:
        raise SystemExit(f"No model at {args.model}. Run record.py, then train.py first.")
    if args.cursor:
        print(move_cursor(mimic, args.to, args.repeat, args.radius, args.temperature, args.pause))
    else:
        CompareDemo(mimic).run()


if __name__ == "__main__":
    main()
