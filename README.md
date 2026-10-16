# Mouse Mimic

Records how *you* move the mouse, trains a neural network on it, and generates new movements in your style. It can draw them on screen or drive the real cursor.

## The app (MouseMimic.exe)
`MouseMimic.exe` is the whole thing in one program; Python is not needed to run it. Build it with `build.bat` (see below), which also copies it to your Desktop. Double-click it and use the four steps in the window: **Record**, **Train**, **Compare**, **Move cursor**. It takes about 10 seconds to start, because it unpacks itself first.

The exe keeps its recordings and model in `%LOCALAPPDATA%\MouseMimic` ("Open folder" in the app). That's separate from this project's `data`/`models` folders.

To rebuild the exe after changing the code, run `build.bat`. It needs `pip install pyinstaller`.

## Running from source
`python app.py` opens the same window. The individual scripts below still work on their own.

## Setup
Python 3.12 is installed. Open a **new** terminal so `python` is on the PATH, then:

```
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## Use it
1. **Record:** `python record.py`. Click the red circles as you normally would; each click saves one move. Esc stops. Aim for 300+ moves (about 10 minutes). Running it again adds to the existing data.
2. **Train:** `python train.py`. Takes a minute or two on the CPU. Saves `models/mimic.pt`.
3. **Compare:** `python demo.py`. Space replays one of your real moves (gray) next to a generated one (orange) at real speed. Click anywhere to generate a path to that point. G draws 10 samples to show the variation. The sliders change temperature (randomness) and target size.
4. **Drive the cursor:** `python demo.py --cursor` (or `--to X Y`, `--repeat N`). It only moves the cursor and never clicks. **Esc or moving the mouse yourself stops it.**

You can also call it from your own code:
```python
from generate import Mimic
path = Mimic().generate_path(start=(100, 100), target=(900, 500), radius=10)  # rows of (t, x, y)
```

## How it works
- **Data:** each move runs from one click to the next. The still "reaction time" at the start is trimmed, then the move is resampled every 8ms, so speed is captured too.
- **Canonical frame:** every move is rotated and scaled so it runs from (0,0) to (1,0). The model then learns the *shape* and *speed profile* of your movements, independent of direction. The real distance and target size go in as conditioning (Fitts' law: far and small targets mean slower, more careful moves).
- **Model:** a 2-layer GRU reads the path so far and outputs a *mixture of 8 Gaussians* for the next step, plus the probability that the move is finished. Generation samples from that distribution one step at a time, which gives natural variation, overshoots and small corrections rather than one perfect curve. This is the same idea as Graves' handwriting generator.
- **Landing:** if the model stops outside the target, a short pause and a quick corrective submovement (minimum-jerk, timed by Fitts' law) take it in, which is what people do after a miss. Any remaining error is smoothed out of the tail, so it always lands inside the target.
- **Getting better results:** more data matters most. If moves come out slower than yours, try `--temperature 0.8`, or train longer (`--epochs 120`).

## Files
| File | |
|---|---|
| `record.py` | target-clicking recorder → `data/moves.jsonl` |
| `data.py` | resampling, canonical frame, features (shared by everything) |
| `model.py` | GRU + mixture density network |
| `train.py` | training (`--epochs`, `--hidden`, `--mirror`, ...) |
| `generate.py` | `Mimic.generate_path(...)` |
| `demo.py` | visual comparison and cursor playback |
