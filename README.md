# A Lightweight Optical Flow Framework for Proactive Crowd Panic Prediction

## Overview
This repository contains a compact, research-oriented prototype for proactive crowd panic prediction using optical flow. The core idea is to compute per-frame motion features and combine them into a Motion Instability Index (MII) that can indicate early signs of crowd panic.

## Contents
- Crowd-Activity-All.avi — A sample video bundled with the repository (used by default).
- crowd_monitor.py — Main script / entry point that processes video frames, computes optical flow features, computes MII, runs a small state machine, visualizes results, and evaluates predictions against ground truth.
- requirements.txt — Minimal dependencies list. Note: scikit-learn is used by the script but not included in requirements.txt by default.

## Quick start
1. Create a Python virtual environment and activate it:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# scikit-learn is required for evaluation metrics
pip install scikit-learn
```

3. Run the monitor:

```bash
python crowd_monitor.py
```

The script opens a window titled "Crowd Monitoring". Press ESC to quit.

Note: On Windows the script uses winsound.Beep for alerts. On other OSes you will need to replace that with a cross-platform audio library or guard the import.

## What crowd_monitor.py does (high level)
- Loads a video (default: `Crowd-Activity-All.avi`).
- Resizes frames to 640×360.
- Computes Farneback optical flow (cv2.calcOpticalFlowFarneback).
- Extracts motion features: speed variance, directional entropy, and simple acceleration.
- Combines features into the Motion Instability Index (MII):
  MII = 0.35 * speed_variance + 0.25 * entropy + 0.4 * acceleration
- Maintains a smoothed MII history and an adaptive threshold.
- Runs a small 3-state machine (NORMAL, EARLY, PANIC) that also checks a 3×3 micro-pattern grid for localized instability.
- Displays a real-time overlay with a heatmap, grid highlights, MII value, and state text.
- Plays an audible beep when panic is detected (Windows-only by default).
- Records predictions and evaluates them against hard-coded ground-truth panic segments, printing a confusion matrix, classification report, accuracy, and plotting the MII trend at the end.

## Configuration / Key variables
- Input video: change the `video_path` variable near the top of `crowd_monitor.py` (previously referenced as line 13 in older versions).
- Ground-truth segments: update `panic_segments` in the script (previously referenced around lines 46–50).
- Thresholds and window sizes: several thresholds and smoothing lengths are hard-coded near the top of the script and inside the MII / state logic. Consider adding CLI args for these.

## Platform notes
- winsound is Windows-only. To run on macOS/Linux, either guard the import or replace it with a cross-platform library such as `playsound` or `simpleaudio`.
- The script uses OpenCV's `cv2.imshow` for visualization. For headless runs (CI or server) switch to saving frames/video instead of showing windows and use offscreen plotting for matplotlib.

## Implementation notes & gotchas
- Hard-coded inputs: The script expects `Crowd-Activity-All.avi` in the repo root by default. Place your own video there or change the input path in the script.
- Ground-truth: `panic_segments` is a small hard-coded list used for offline evaluation. Update or remove as needed for your dataset.
- Missing dependency: scikit-learn is used for metrics — add it to `requirements.txt` if you want to reproduce the evaluation output.
- No CLI: Parameters (video path, thresholds, grid size, smoothing lengths) are embedded in the script. Converting to argparse would make the script more reusable.

## Suggested improvements
- Add argparse to allow setting input video, thresholds, and output options from the command line.
- Replace winsound with a cross-platform notifier or make audio alerts optional.
- Move MII computation and the state machine logic into a small module so the logic can be unit-tested.
- Add a short Jupyter notebook that steps through a sample video frame-by-frame and explains MII components visually.
- Add scikit-learn to `requirements.txt` and pin versions for reproducibility.
