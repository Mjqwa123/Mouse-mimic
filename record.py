"""Record your mouse movements by clicking targets.

Circles appear one at a time; move to each and click it. Every click on a
target saves the path from the previous click. Esc finishes. Sessions append
to data/moves.jsonl, so you can record in several sittings.

    python record.py
"""
import json
import math
import random
import time
import tkinter as tk

from data import DATA_PATH, make_dpi_aware

MIN_RADIUS, MAX_RADIUS = 10, 60
MAX_DURATION = 4.0  # seconds; slower moves are probably distractions, not aiming


class Recorder:
    def __init__(self, out_path=DATA_PATH, master=None):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.master = master
        self.root = tk.Toplevel(master) if master else tk.Tk()
        self.root.title("Mouse Mimic - recorder")
        self.root.attributes("-fullscreen", True)
        self.canvas = tk.Canvas(self.root, bg="#15171c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()
        self.w, self.h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()

        self.points = []
        self.start = None
        self.t_start = 0.0
        self.misses = 0
        self.saved = 0
        self.target = (self.w / 2, self.h / 2)
        self.radius = 30

        self.info = self.canvas.create_text(20, 20, anchor="nw", fill="#c8ccd4", font=("Segoe UI", 14), text="")
        self.circle = None
        self.draw_target()
        self.update_info("Click the circle to start. Esc to finish.")

        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def update_info(self, extra=""):
        self.canvas.itemconfigure(
            self.info, text=f"Moves recorded this session: {self.saved}   (300+ recommended)\n{extra}")

    def draw_target(self):
        if self.circle is not None:
            self.canvas.delete(self.circle)
        ox, oy = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        x, y, r = self.target[0] - ox, self.target[1] - oy, self.radius
        self.circle = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#e0644f", outline="#ffb199", width=2)

    def new_target(self, x, y):
        self.radius = round(math.exp(random.uniform(math.log(MIN_RADIUS), math.log(MAX_RADIUS))))
        m = self.radius + 20
        # a third of moves use a log-uniform distance so short moves are well covered too
        near = random.random() < 0.33
        dist_goal = math.exp(random.uniform(math.log(60), math.log(450)))
        for _ in range(200):
            if near:
                a = random.uniform(0, 2 * math.pi)
                tx, ty = x + dist_goal * math.cos(a), y + dist_goal * math.sin(a)
            else:
                tx, ty = random.uniform(m, self.w - m), random.uniform(m + 60, self.h - m)
            if m <= tx <= self.w - m and m + 60 <= ty <= self.h - m and math.hypot(tx - x, ty - y) >= 60:
                break
        self.target = (min(max(tx, m), self.w - m), min(max(ty, m + 60), self.h - m))
        self.draw_target()

    def on_motion(self, e):
        if self.start is not None:
            self.points.append([round(time.perf_counter(), 6), e.x_root, e.y_root])

    def on_click(self, e):
        now = time.perf_counter()
        x, y = e.x_root, e.y_root
        if math.hypot(x - self.target[0], y - self.target[1]) > self.radius:
            self.misses += 1
            return
        if self.start is not None:
            pts = [[round(self.t_start, 6), *self.start]] + self.points + [[round(now, 6), x, y]]
            if now - self.t_start <= MAX_DURATION and len(pts) >= 4:
                move = {
                    "start": list(self.start),
                    "target": [round(self.target[0], 1), round(self.target[1], 1)],
                    "radius": self.radius,
                    "misses": self.misses,
                    "screen": [self.w, self.h],
                    "points": pts,
                }
                with open(self.out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(move) + "\n")
                self.saved += 1
                self.update_info("Esc to finish.")
            else:
                self.update_info("(that one was too slow, skipped)  Esc to finish.")
        self.start = (x, y)
        self.t_start = now
        self.points = []
        self.misses = 0
        self.new_target(x, y)

    def run(self):
        """Show the recorder until Esc; returns the number of moves saved."""
        if self.master:
            self.root.focus_force()
            self.master.wait_window(self.root)
        else:
            self.root.mainloop()
        return self.saved


if __name__ == "__main__":
    make_dpi_aware()
    n = Recorder().run()
    print(f"Saved {n} moves to {DATA_PATH}")
