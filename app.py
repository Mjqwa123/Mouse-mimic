"""Mouse Mimic: one window for recording, training, comparing and cursor playback.

    python app.py
"""
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

from data import DATA_PATH, MODEL_PATH, ROOT, make_dpi_aware

BG, PANEL, FG, MUTED, ACCENT = "#15171c", "#1f2229", "#d6d9e0", "#8a8f99", "#ff9f43"


def resource(name):
    """Path to a file bundled with the app (works both from source and from the .exe)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mouse Mimic")
        self.root.configure(bg=BG)
        self.root.minsize(560, 600)
        h = min(800, self.root.winfo_screenheight() - 90)
        self.root.geometry(f"600x{h}+{(self.root.winfo_screenwidth() - 600) // 2}+20")
        try:
            self.root.iconbitmap(default=resource("icon.ico"))
        except tk.TclError:
            pass

        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        self.busy = None  # "train" or "cursor" while a background job runs
        self.mimic = None
        self.mimic_mtime = None
        # torch takes a few seconds to import; start that now so the first click is quick
        threading.Thread(target=self._preload, daemon=True).start()

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=BG, background=ACCENT, bordercolor=PANEL,
                        lightcolor=ACCENT, darkcolor=ACCENT)

        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="both", expand=True, padx=18, pady=10)
        tk.Label(wrap, text="Mouse Mimic", bg=BG, fg=FG, font=("Segoe UI Semibold", 20)).pack(anchor="w")
        self.status = tk.Label(wrap, text="", bg=BG, fg=MUTED, font=("Segoe UI", 10), justify="left")
        self.status.pack(anchor="w", pady=(0, 6))

        # 1. Record
        box = self.section(wrap, "1  Record", "Click the red circles as you normally would. Esc when done. "
                                              "300+ moves recommended; more sessions add to what you have.")
        self.record_btn = self.button(box, "Record moves", self.record)

        # 2. Train
        box = self.section(wrap, "2  Train", "Teaches the model your movements. Takes a few minutes.")
        row = tk.Frame(box, bg=PANEL)
        row.pack(fill="x")
        self.train_btn = self.button(row, "Train", self.train, side="left")
        tk.Label(row, text="rounds", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left", padx=(14, 4))
        self.epochs = tk.Spinbox(row, from_=5, to=500, increment=5, width=5, bg=BG, fg=FG, buttonbackground=PANEL,
                                 insertbackground=FG, relief="flat", font=("Segoe UI", 10))
        self.epochs.delete(0, "end")
        self.epochs.insert(0, "60")
        self.epochs.pack(side="left")
        self.progress = ttk.Progressbar(box, mode="determinate", maximum=60)
        self.progress.pack(fill="x", pady=(8, 0))

        # 3. Compare
        box = self.section(wrap, "3  Compare", "Space: your real move vs a generated one.  Click: generate to "
                                               "that point.  G: 10 samples.  Esc: back.")
        self.compare_btn = self.button(box, "Open compare view", self.compare)

        # 4. Cursor
        box = self.section(wrap, "4  Move the real cursor", "Moves your cursor in your style. It never clicks. "
                                                            "Press Esc or move your mouse to stop.")
        row = tk.Frame(box, bg=PANEL)
        row.pack(fill="x")
        self.cursor_btn = self.button(row, "Move cursor", self.cursor, side="left")
        tk.Label(row, text="moves", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left", padx=(14, 4))
        self.repeat = tk.Spinbox(row, from_=1, to=100, width=4, bg=BG, fg=FG, buttonbackground=PANEL,
                                 insertbackground=FG, relief="flat", font=("Segoe UI", 10))
        self.repeat.delete(0, "end")
        self.repeat.insert(0, "5")
        self.repeat.pack(side="left")
        tk.Label(row, text="randomness", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left", padx=(14, 4))
        self.temp = tk.DoubleVar(value=1.0)
        tk.Scale(row, variable=self.temp, from_=0.3, to=1.5, resolution=0.05, orient="horizontal", length=130,
                 bg=PANEL, fg=FG, highlightthickness=0, troughcolor=BG, showvalue=True,
                 font=("Segoe UI", 8)).pack(side="left")

        # footer first, packed to the bottom, so the log is what shrinks on small screens
        foot = tk.Frame(wrap, bg=BG)
        foot.pack(side="bottom", fill="x")
        tk.Label(foot, text=f"Data: {ROOT}", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
        tk.Button(foot, text="Open folder", command=self.open_folder, bg=BG, fg=ACCENT, relief="flat", bd=0,
                  activebackground=BG, activeforeground=FG, cursor="hand2", font=("Segoe UI", 9)).pack(side="right")

        # log
        self.log_box = tk.Text(wrap, height=4, bg=PANEL, fg=FG, relief="flat", font=("Consolas", 9),
                               state="disabled", wrap="word", padx=8, pady=6)
        self.log_box.pack(fill="both", expand=True, pady=(8, 4))

        self.log("Ready." if MODEL_PATH.exists() else "Welcome! Start with step 1: record some moves.")
        self.refresh()
        self.root.after(100, self.poll)

    # ---- layout helpers
    def section(self, parent, title, text):
        box = tk.Frame(parent, bg=PANEL, padx=14, pady=8)
        box.pack(fill="x", pady=4)
        tk.Label(box, text=title, bg=PANEL, fg=FG, font=("Segoe UI Semibold", 12)).pack(anchor="w")
        tk.Label(box, text=text, bg=PANEL, fg=MUTED, font=("Segoe UI", 9), wraplength=520,
                 justify="left").pack(anchor="w", pady=(0, 6))
        return box

    def button(self, parent, text, command, side=None):
        b = tk.Button(parent, text=text, command=command, bg=ACCENT, fg="#1a1a1a", activebackground="#ffb46b",
                      relief="flat", bd=0, padx=14, pady=5, cursor="hand2", font=("Segoe UI Semibold", 10))
        b.pack(side=side, anchor="w") if side else b.pack(anchor="w")
        return b

    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ---- state
    def _preload(self):
        try:
            import generate  # noqa: F401
            import train  # noqa: F401
        except Exception:
            pass

    def count_moves(self):
        try:
            with open(DATA_PATH, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    def refresh(self):
        n = self.count_moves()
        if MODEL_PATH.exists():
            when = time.strftime("%b %d, %H:%M", time.localtime(MODEL_PATH.stat().st_mtime))
            model = f"trained {when}"
        else:
            model = "not trained yet"
        self.status.configure(text=f"Recorded moves: {n}      Model: {model}")
        idle = self.busy is None
        for b in (self.record_btn, self.compare_btn):
            b.configure(state="normal" if idle else "disabled")
        self.train_btn.configure(text="Stop" if self.busy == "train" else "Train",
                                 state="normal" if self.busy in (None, "train") else "disabled")
        self.cursor_btn.configure(text="Stop" if self.busy == "cursor" else "Move cursor",
                                  state="normal" if self.busy in (None, "cursor") else "disabled")

    def get_mimic(self):
        """Load the model, reloading it if it was retrained since last time."""
        if not MODEL_PATH.exists():
            self.log("No model yet. Record some moves and press Train first.")
            return None
        mtime = MODEL_PATH.stat().st_mtime
        if self.mimic is None or mtime != self.mimic_mtime:
            from generate import Mimic
            self.mimic = Mimic(MODEL_PATH)
            self.mimic_mtime = mtime
        return self.mimic

    def poll(self):
        """Apply messages from background jobs on the Tk thread."""
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log(value)
                    if value.startswith("epoch"):
                        self.progress["value"] = int(value.split()[1])
                elif kind == "done":
                    self.busy = None
                    self.log(value)
                    self.refresh()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def run_job(self, kind, fn):
        self.busy = kind
        self.stop_flag.clear()
        self.refresh()

        def worker():
            try:
                msg = fn()
            except Exception as e:  # report instead of dying silently in the thread
                msg = f"Error: {e}"
            self.events.put(("done", msg))

        threading.Thread(target=worker, daemon=True).start()

    # ---- actions
    def record(self):
        from record import Recorder
        n = Recorder(DATA_PATH, master=self.root).run()
        self.log(f"Recorded {n} new moves.")
        self.refresh()

    def train(self):
        if self.busy == "train":
            self.stop_flag.set()
            self.log("Stopping after this round...")
            return
        try:
            epochs = max(1, int(self.epochs.get()))
        except ValueError:
            epochs = 60
        self.progress.configure(maximum=epochs, value=0)
        self.log(f"Training for {epochs} rounds...")

        def job():
            from train import train
            best = train(DATA_PATH, MODEL_PATH, epochs=epochs,
                         log=lambda m: self.events.put(("log", m)), should_stop=self.stop_flag.is_set)
            return "Training finished." if best != float("inf") else "Training stopped before a model was saved."

        def safe_job():
            try:
                return job()
            except ValueError as e:
                return str(e)

        self.run_job("train", safe_job)

    def compare(self):
        mimic = self.get_mimic()
        if mimic is None:
            return
        from demo import CompareDemo
        CompareDemo(mimic, master=self.root).run()

    def cursor(self):
        if self.busy == "cursor":
            self.stop_flag.set()
            return
        mimic = self.get_mimic()
        if mimic is None:
            return
        try:
            repeat = max(1, int(self.repeat.get()))
        except ValueError:
            repeat = 5
        temperature = self.temp.get()
        from demo import move_cursor
        self.run_job("cursor", lambda: move_cursor(
            mimic, repeat=repeat, temperature=temperature,
            log=lambda m: self.events.put(("log", m)), should_stop=self.stop_flag.is_set))

    def open_folder(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        os.startfile(ROOT)

    def run(self):
        try:
            import pyi_splash  # only exists inside the packaged .exe
            pyi_splash.close()
        except ImportError:
            pass
        self.root.mainloop()


if __name__ == "__main__":
    make_dpi_aware()
    App().run()
