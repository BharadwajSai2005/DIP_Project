## A Lightweight Optical Flow Framework for Proactive Crowd Panic Prediction Using Motion Instability Index

## About this repository
Hi — I built this repository as a compact, research-oriented prototype to explore proactive crowd panic prediction using optical flow. The core idea is simple: compute per-frame motion features (from Farneback optical flow), aggregate them into a Motion Instability Index (MII), and use a small state machine to flag EARLY / PANIC conditions. This is meant for experimentation and prototyping, not for production or safety-critical deployments.

## What’s in this repo (file-by-file)
Crowd-Activity-All.avi

A sample video bundled with the repo that the script uses by default. Place your own test video here or change the script input path.

crowd_monitor.py

The single main script / entry point. It:
Loads the video (hard-coded to "Crowd-Activity-All.avi"),
Resizes frames to 640×360,
Computes Farneback optical flow (cv2.calcOpticalFlowFarneback),
Extracts motion features: speed variance, directional entropy, and simple acceleration,
Combines them into an MII formula: MII = 0.35speed_variance + 0.25entropy + 0.4*acceleration,
Maintains a smoothed MII history and an adaptive threshold,
Runs a 3-state machine (NORMAL, EARLY, PANIC) that also checks a 3×3 micro-pattern grid for localized instability,
Shows a real-time overlay window with heatmap, bounding-grid highlights, MII and state text,
Plays a beep (using Windows winsound) when panic is detected,
Records predictions and evaluates them against hard-coded ground-truth panic segments,
Prints confusion matrix, classification report, accuracy, and plots the MII trend at the end.
Key locations to change:
Line 13: video filename / path
Lines 46–50: panic_segments (the ground-truth time windows used for evaluation)
Several hard-coded thresholds and window sizes are near the top of the script and in the MII / state logic.
requirements.txt

Minimal list: opencv-python, numpy, matplotlib, ipython, ipywidgets
Note: the script imports scikit-learn (sklearn.metrics) but scikit-learn is not currently included in requirements.txt — add it before running evaluation code.
Quick start (how I run it locally)
Create a virtual environment and install dependencies:

## Code
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install scikit-learn       # required for evaluation output
Run the monitor:
Code
python crowd_monitor.py
The script opens a window titled "Crowd Monitoring". Press ESC to quit.
On Windows the script uses winsound.Beep for alerts. On other OSes you will need to replace that with a cross-platform audio library or guard the import.
Implementation notes and gotchas (things I want you to know)
Hard-coded inputs: The script expects a file named Crowd-Activity-All.avi in the repo root. If you use your own video place it there or edit line 13.
Ground-truth: panic_segments is a small hard-coded list of time windows used for offline evaluation. Update or remove as needed for your dataset.
Platform-specific code: winsound is Windows-only. If you run on Linux/macOS, wrap the import or replace with playsound/simpleaudio.
Missing dependency in requirements.txt: scikit-learn is used for metrics — add it to requirements.txt if you want to reproduce the printed evaluation.
No CLI/arguments: Parameters (video path, thresholds, grid size, smoothing lengths) are embedded in the script. I kept it simple for experimentation; converting to argparse would make it reusable.
Visualization: The script uses cv2.imshow for interactive display and matplotlib for a final plot. To run headless (CI or server), switch to saving frames/video rather than showing windows and use opencv-python-headless.

## Suggested improvements (if I continue work on this)
Add argparse to set input video, thresholds, and output options from the command line.
Replace winsound with a cross-platform notifier or make audio optional.
Move MII computation and state machine into a small module so I can unit-test the logic.
Add a short notebook that steps through a sample video frame-by-frame and explains the MII components visually.
Add scikit-learn to requirements.txt and pin versions for reproducibility.
