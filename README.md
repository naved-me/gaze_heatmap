# Real-Time Gaze Heatmap

A cross-platform (Windows/Linux/macOS) real-time eye-tracking and gaze heatmap application. It tracks your iris position and nose tip using a webcam, calibrates your gaze to the screen corners, and displays an accumulating, fading Gaussian heatmap directly on your screen.

## Features

- **Real-Time Iris Tracking:** Uses Google's MediaPipe Face Mesh / FaceLandmarker models for robust eye and face tracking.
- **Screen Calibration:** Easy 4-point corner calibration to map your gaze to your screen resolution.
- **Live Gaze Heatmap:** Displays an interactive, fading heatmap showing exactly where you are looking in real-time.
- **Cross-Platform Overlay:** 
  - On Windows: The heatmap runs as a transparent, click-through, always-on-top overlay over your desktop.
  - On Linux/macOS: Displays in a standard OpenCV window.
- **Session Heatmap Aggregation:** Track and save the *entire* session's gaze history! You can accumulate all gaze points into a single image to see a complete map of where you looked over the duration of the session.

## Prerequisites

- Python 3.8+
- Webcam

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
*(Dependencies generally include `opencv-python`, `numpy`, and `mediapipe`)*

## Usage

To start the real-time tracker:
```bash
python gaze_heatmap.py
```

### Saving the Final Aggregated Heatmap

If you want to save a final combined image of everywhere you looked during the session (without affecting the live fading preview), run the script with the `--save-heatmap` flag:

```bash
python gaze_heatmap.py --save-heatmap
```
When you exit the application (by pressing `ESC`), this will save the merged heatmap as `final_heatmap.png` in the project directory.

To specify a custom output filename:
```bash
python gaze_heatmap.py --save-heatmap --output my_gaze_results.jpg
```

### Advanced Arguments

- `--camera`: Webcam index (default: `0`)
- `--width` / `--height`: Screen resolution for mapping (default: `1920x1080`)
- `--sigma`: Gaussian blur size for the heatmap blob (default: `18.0`)
- `--fade`: How quickly the live heatmap fades (0-1). Lower is faster. (default: `0.85`)
- `--no-overlay`: Disables the transparent Windows overlay and uses a standard window.
- `--head-suppression`: Controls how much head movement influences gaze. (default: `1.0` - full range)

## Calibration Process

When the application starts, it will enter calibration mode:
1. Look at the red dot displayed on your screen.
2. Press the `c` key to capture samples.
3. Repeat for all calibration points (typically the four corners of your screen).
4. Once completed, the main heatmap will launch automatically! Press `ESC` at any time to quit.