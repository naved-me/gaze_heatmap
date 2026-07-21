"""
Real-Time Gaze Heatmap - Cross-Platform (Windows/Linux/macOS)
Tracks iris position + nose tip, calibrates to screen corners, and displays
an accumulating Gaussian heatmap overlay.

Supports both MediaPipe APIs:
  - mp.solutions.face_mesh (legacy, with refine_landmarks=True for iris)
  - mediapipe.tasks.vision.FaceLandmarker (newer API, downloads model)

Windows: Transparent, click-through, always-on-top overlay.
Linux/macOS: Standard OpenCV window (no transparency due to API limitations).
"""

import argparse
import sys
import time
import os
import platform

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe auto-detection: try solutions first, fall back to tasks
# ---------------------------------------------------------------------------
MP_API = None  # 'solutions' or 'tasks'
mp = None

# Try legacy solutions API first (better iris support via refine_landmarks)
try:
    import mediapipe as _mp
    if hasattr(_mp, "solutions") and hasattr(_mp.solutions, "face_mesh"):
        mp = _mp
        MP_API = "solutions"
        print("[INFO] Using MediaPipe solutions API (mp.solutions.face_mesh)")
except Exception:
    pass

# Fall back to tasks API
if MP_API is None:
    try:
        import mediapipe as _mp
        mp = _mp
        MP_API = "tasks"
        print("[INFO] Using MediaPipe tasks API (FaceLandmarker)")
    except Exception as e:
        raise RuntimeError("Failed to import mediapipe: %r" % (e,))

if mp is None:
    raise RuntimeError("MediaPipe is required but could not be imported.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def compute_pupil_center(landmarks_xy, idxs, drop_ends=False):
    """
    Compute a stable pupil/iris center from iris landmarks.
    If drop_ends=True and we have exactly 5 iris points, uses only the middle 3
    to reduce noise/jitter from the outer iris points.
    """
    pts = landmarks_xy[idxs]
    if drop_ends and len(idxs) == 5:
        pts = pts[1:4]  # drop first and last -> keep middle 3
    return pts.mean(axis=0)


def gaussian_blob(shape, center_xy, sigma_px):
    h, w = shape
    x0, y0 = center_xy
    if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
        return None
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma_px ** 2))
    return g


def draw_circle(img, xy, color, r=4, thickness=-1):
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(img, (x, y), r, color, thickness)


# ---------------------------------------------------------------------------
# Windows overlay helpers (ctypes)
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import ctypes
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    GWL_EXSTYLE = -20
    HWND_TOPMOST = -1
    LWA_COLORKEY = 0x00000001
    user32 = ctypes.windll.user32
else:
    user32 = None


def setup_windows_overlay(window_name):
    """Make OpenCV window transparent, click-through, and topmost on Windows."""
    if not IS_WINDOWS:
        return None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    hwnd = None
    for _ in range(50):
        hwnd_try = user32.FindWindowW(None, window_name)
        if hwnd_try:
            hwnd = hwnd_try
            break
        time.sleep(0.02)

    if hwnd:
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        style |= WS_EX_LAYERED
        style |= WS_EX_TRANSPARENT
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, LWA_COLORKEY)
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, 0x0002 | 0x0001)

    return hwnd


# ---------------------------------------------------------------------------
# Landmark detector abstraction
# ---------------------------------------------------------------------------
class FaceMeshDetector:
    """Wraps either mp.solutions.face_mesh or mp.tasks.vision.FaceLandmarker."""

    def __init__(self, api_type):
        self.api_type = api_type
        self.detector = None
        self._mp_image = None
        self._mp_image_format = None

        if api_type == "solutions":
            self._init_solutions()
        else:
            self._init_tasks()

    def _init_solutions(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.detector = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _init_tasks(self):
        import urllib.request
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "face_landmarker.task")

        if not (os.path.exists(model_path) and os.path.getsize(model_path) > 0):
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            print(f"[INFO] Downloading FaceLandmarker model to: {model_path}")
            urllib.request.urlretrieve(model_url, model_path)
            print("[INFO] Download complete.")

        base_options = BaseOptions(model_asset_path=model_path)
        options_kwargs = {
            "base_options": base_options,
            "num_faces": 1,
            "output_face_blendshapes": False,
        }
        try:
            from mediapipe.tasks.python.vision import VisionRunningMode
            options_kwargs["running_mode"] = VisionRunningMode.IMAGE
        except Exception:
            pass

        face_landmarker_options = FaceLandmarkerOptions(**options_kwargs)
        self.detector = FaceLandmarker.create_from_options(face_landmarker_options)
        self._mp_image = mp.Image
        self._mp_image_format = getattr(mp, "ImageFormat", None)

    def detect(self, bgr_frame):
        if self.api_type == "solutions":
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            res = self.detector.process(rgb)
            if not res or not res.multi_face_landmarks:
                return None
            return res.multi_face_landmarks[0]
        else:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            img_fmt = None
            if self._mp_image_format is not None:
                for candidate in ("SRGB", "SRGBColor", "SRGBA", "SRG"):
                    if hasattr(self._mp_image_format, candidate):
                        img_fmt = getattr(self._mp_image_format, candidate)
                        break

            if img_fmt is None:
                image = self._mp_image(data=rgb)
            else:
                image = self._mp_image(image_format=img_fmt, data=rgb)

            detection_result = self.detector.detect(image)
            if detection_result is None or not detection_result.face_landmarks:
                return None
            return detection_result.face_landmarks[0]


# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
NOSE_TIP = 4
LEFT_IRIS_IDXS = [473, 474, 475, 476, 477]
RIGHT_IRIS_IDXS = [468, 469, 470, 471, 472]
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (362, 263)


# ---------------------------------------------------------------------------
# Gaze vector extraction
# ---------------------------------------------------------------------------
def extract_gaze_vector(landmarks, frame_w, frame_h, api_type, iris_drop_ends=False):
    """
    Convert landmarks to a normalized (u, v) gaze vector.
    Uses 2-point pupil centers only (no nose/head bias).
    """
    if api_type == "solutions":
        pts = np.array(
            [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks.landmark],
            dtype=np.float32,
        )
    else:
        pts = np.array(
            [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks],
            dtype=np.float32,
        )

    left_pupil = compute_pupil_center(pts, LEFT_IRIS_IDXS, drop_ends=iris_drop_ends)
    right_pupil = compute_pupil_center(pts, RIGHT_IRIS_IDXS, drop_ends=iris_drop_ends)

    u = float((left_pupil[0] + right_pupil[0]) / (2.0 * frame_w))
    v = float((left_pupil[1] + right_pupil[1]) / (2.0 * frame_h))

    u = clamp(u, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)

    return np.array([u, v], dtype=np.float32), pts


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_landmarks_on_frame(frame, pts):
    """Draw nose, pupils, and eye corners on the webcam frame."""
    nose = tuple(pts[NOSE_TIP].astype(int))
    left_pupil = compute_pupil_center(pts, LEFT_IRIS_IDXS)
    right_pupil = compute_pupil_center(pts, RIGHT_IRIS_IDXS)
    l1, l2 = LEFT_CORNERS
    r1, r2 = RIGHT_CORNERS

    cv2.circle(frame, nose, 5, (255, 0, 0), -1)
    cv2.circle(frame, tuple(left_pupil.astype(int)), 5, (0, 255, 0), -1)
    cv2.circle(frame, tuple(right_pupil.astype(int)), 5, (0, 255, 0), -1)
    cv2.circle(frame, tuple(pts[l1].astype(int)), 4, (255, 255, 0), -1)
    cv2.circle(frame, tuple(pts[l2].astype(int)), 4, (255, 255, 0), -1)
    cv2.circle(frame, tuple(pts[r1].astype(int)), 4, (255, 255, 0), -1)
    cv2.circle(frame, tuple(pts[r2].astype(int)), 4, (255, 255, 0), -1)


def draw_calibration_corner(img, cx, cy, W, H, corner_name):
    """Draw a corner guide for calibration."""
    if corner_name == "TL":
        cv2.line(img, (cx, cy), (0, 0), (255, 255, 255), 2)
        cv2.line(img, (cx, 0), (cx, cy), (255, 255, 255), 1)
        cv2.line(img, (0, cy), (cx, cy), (255, 255, 255), 1)
    elif corner_name == "TR":
        cv2.line(img, (cx, cy), (W - 1, 0), (255, 255, 255), 2)
        cv2.line(img, (cx, 0), (cx, cy), (255, 255, 255), 1)
        cv2.line(img, (cx, cy), (W - 1, cy), (255, 255, 255), 1)
    elif corner_name == "BL":
        cv2.line(img, (cx, cy), (0, H - 1), (255, 255, 255), 2)
        cv2.line(img, (0, cy), (cx, cy), (255, 255, 255), 1)
        cv2.line(img, (cx, cy), (cx, H - 1), (255, 255, 255), 1)
    elif corner_name == "BR":
        cv2.line(img, (cx, cy), (W - 1, H - 1), (255, 255, 255), 2)
        cv2.line(img, (cx, cy), (W - 1, cy), (255, 255, 255), 1)
        cv2.line(img, (cx, cy), (cx, H - 1), (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calibrated real-time gaze heatmap using MediaPipe Face Mesh."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=1920, help="Screen width for mapping")
    parser.add_argument("--height", type=int, default=1080, help="Screen height for mapping")
    parser.add_argument("--sigma", type=float, default=18.0, help="Gaussian sigma in heatmap pixels")
    parser.add_argument("--fade", type=float, default=0.85, help="Heatmap fading factor per frame (0-1). Lower = faster fade.")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable transparent overlay (show standard window)")
    parser.add_argument("--smooth-alpha", type=float, default=0.12,
                        help="EMA smoothing factor for mapped screen coords (x,y).")
    parser.add_argument("--calibration-mode", type=str, default="corners", choices=["grid", "corners"],
                        help="Calibration type: 'grid' (12 fixed dots) or 'corners' (4 screen corners).")
    parser.add_argument("--calib-samples", type=int, default=40,
                        help="Number of gaze samples to capture per calibration point.")
    parser.add_argument("--calib-method", type=str, default="median", choices=["mean", "median"],
                        help="How to combine calibration samples: mean or median.")
    parser.add_argument("--heat-bg-thresh", type=float, default=0.01,
                        help="Raw heat threshold below which pixels are forced to 0/black.")
    parser.add_argument("--iris-drop-ends", type=int, default=1,
                        help="Use only the middle 3 of the 5 iris landmarks (1=yes, 0=no).")
    parser.add_argument("--head-suppression", type=float, default=1.0,
                        help="Suppress head movement influence on gaze (0.0=no head influence (locked to center), 1.0=full head influence (raw gaze). Default 1.0 = full range.")
    parser.add_argument("--flip-x", type=int, default=0, help="Flip mapped x coordinate (1=yes).")
    parser.add_argument("--flip-y", type=int, default=0, help="Flip mapped y coordinate (1=yes).")
    parser.add_argument("--smooth-mode", type=str, default="median", choices=["median", "ema"],
                        help="Display smoothing mode: median or ema.")
    parser.add_argument("--smooth-window", type=int, default=11,
                        help="Median filter window size for display smoothing.")
    parser.add_argument("--jump-px", type=float, default=30.0,
                        help="Outlier rejection threshold in pixels for display smoothing.")
    parser.add_argument("--save-heatmap", action="store_true", help="Save the final combined heatmap as an image.")
    parser.add_argument("--output", type=str, default="final_heatmap.png", help="Output image filename.")

    args = parser.parse_args()

    W, H = args.width, args.height
    iris_drop_ends = bool(int(args.iris_drop_ends))
    head_suppression = float(args.head_suppression)
    head_suppression = max(0.0, min(1.0, head_suppression))

    detector = FaceMeshDetector(MP_API)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    calib_mode = str(args.calibration_mode).lower()
    dot_margin_px = 50

    if calib_mode == "grid":
        GRID_COLS = 4
        GRID_ROWS = 3
        xs = np.linspace(dot_margin_px, float(W - 1 - dot_margin_px), GRID_COLS, dtype=np.float32)
        ys = np.linspace(dot_margin_px, float(H - 1 - dot_margin_px), GRID_ROWS, dtype=np.float32)
        calib_point_names = []
        screen_points = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                name = f"P{r}_{c}"
                calib_point_names.append(name)
                screen_points.append((float(xs[c]), float(ys[r])))
    else:
        corners_screen = {
            "TL": (dot_margin_px, dot_margin_px),
            "TR": (W - 1 - dot_margin_px, dot_margin_px),
            "BL": (dot_margin_px, H - 1 - dot_margin_px),
            "BR": (W - 1 - dot_margin_px, H - 1 - dot_margin_px),
        }
        calib_point_names = ["TL", "TR", "BL", "BR"]
        screen_points = [corners_screen[name] for name in calib_point_names]

    screen_points = np.array(screen_points, dtype=np.float32)

    # ================================================================
    # CALIBRATION PHASE
    # ================================================================
    print("\n" + "=" * 60)
    print("CALIBRATION MODE")
    print("=" * 60)
    print("Look at each red dot on the screen and press 'c' to capture.")
    print("Press ESC at any time to abort.\n")

    calib_vectors = {}

    if calib_mode == "corners":
        calib_window = "Calibration - Look at the SCREEN CORNER"
    else:
        calib_window = "Calibration - Look at the dot"
    cv2.namedWindow(calib_window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(calib_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    preview_window = "Webcam Preview"
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_window, 400, 300)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[0], frame.shape[1]
        landmarks = detector.detect(frame)

        next_point_idx = len(calib_vectors)
        next_point_name = calib_point_names[next_point_idx] if next_point_idx < len(calib_point_names) else None

        if next_point_name:
            calib_screen = np.zeros((H, W, 3), dtype=np.uint8)

            idx = next_point_idx
            cx, cy = screen_points[idx]
            cx_i, cy_i = int(cx), int(cy)

            if calib_mode == "corners":
                cv2.circle(calib_screen, (cx_i, cy_i), 25, (0, 0, 255), -1)
                cv2.circle(calib_screen, (cx_i, cy_i), 30, (255, 255, 255), 2)
                corner_name = next_point_name
                draw_calibration_corner(calib_screen, cx_i, cy_i, W, H, corner_name)

                corner_name_full = {"TL": "Top-Left", "TR": "Top-Right",
                                    "BL": "Bottom-Left", "BR": "Bottom-Right"}[corner_name]
                text = f"Look at the RED DOT in the {corner_name_full} corner of your screen"
            else:
                cv2.circle(calib_screen, (cx_i, cy_i), 20, (0, 0, 255), -1)
                cv2.circle(calib_screen, (cx_i, cy_i), 25, (255, 255, 255), 2)
                text = f"Look at the RED DOT (sample {idx+1}/{len(calib_point_names)})"

            cv2.putText(calib_screen, text, (W // 2 - 460, H // 2 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(calib_screen, "Press 'c' to capture  |  ESC to abort",
                        (W // 2 - 300, H // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(calib_screen, f"Captured: {len(calib_vectors)}/{len(calib_point_names)}",
                        (W // 2 - 140, H // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            calib_screen = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(calib_screen, "Calibration Complete! Starting...",
                        (W // 2 - 300, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow(calib_window, calib_screen)

        preview = frame.copy()
        vec = None
        pts = None

        if landmarks is not None:
            vec, pts = extract_gaze_vector(landmarks, frame_w, frame_h, MP_API, iris_drop_ends=iris_drop_ends)
            draw_landmarks_on_frame(preview, pts)
            cv2.putText(preview, f"u={vec[0]:.3f} v={vec[1]:.3f}",
                        (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(preview, f"Captured: {len(calib_vectors)}/{len(calib_point_names)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if next_point_name:
            cv2.putText(preview, f"Next: {next_point_name} - press 'c'",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(preview_window, preview)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Calibration aborted.")
            return

        if key == ord("c") and next_point_name is not None and vec is not None:
            samples_u = [float(vec[0])]
            samples_v = [float(vec[1])]

            target = max(1, int(args.calib_samples))

            idx = next_point_idx
            cx, cy = screen_points[idx]
            cx_i, cy_i = int(cx), int(cy)

            while len(samples_u) < target:
                ret2, frame2 = cap.read()
                if not ret2:
                    continue

                frame_h2, frame_w2 = frame2.shape[0], frame2.shape[1]
                landmarks2 = detector.detect(frame2)
                vec2 = None
                if landmarks2 is not None:
                    vec2, pts2 = extract_gaze_vector(
                        landmarks2, frame_w2, frame_h2, MP_API, iris_drop_ends=iris_drop_ends,
                    )

                calib_screen_sample = np.zeros((H, W, 3), dtype=np.uint8)
                if calib_mode == "corners":
                    cv2.circle(calib_screen_sample, (cx_i, cy_i), 25, (0, 0, 255), -1)
                    cv2.circle(calib_screen_sample, (cx_i, cy_i), 30, (255, 255, 255), 2)
                    corner_name = next_point_name
                    draw_calibration_corner(calib_screen_sample, cx_i, cy_i, W, H, corner_name)
                else:
                    cv2.circle(calib_screen_sample, (cx_i, cy_i), 20, (0, 0, 255), -1)
                    cv2.circle(calib_screen_sample, (cx_i, cy_i), 25, (255, 255, 255), 2)

                cv2.putText(calib_screen_sample,
                            f"Capturing {next_point_name}: {len(samples_u)}/{target}",
                            (W // 2 - 360, H // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                preview2 = frame2.copy()
                if vec2 is not None and landmarks2 is not None:
                    draw_landmarks_on_frame(preview2, pts2)
                    cv2.putText(preview2, f"u={vec2[0]:.3f} v={vec2[1]:.3f}",
                                (10, frame_h2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow(calib_window, calib_screen_sample)
                cv2.imshow(preview_window, preview2)

                k2 = cv2.waitKey(1) & 0xFF
                if k2 == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Calibration aborted.")
                    return

                if vec2 is not None:
                    samples_u.append(float(vec2[0]))
                    samples_v.append(float(vec2[1]))

            u_samples = np.array(samples_u, dtype=np.float32)
            v_samples = np.array(samples_v, dtype=np.float32)

            if args.calib_method == "median":
                u_final = float(np.median(u_samples))
                v_final = float(np.median(v_samples))
            else:
                u_final = float(u_samples.mean())
                v_final = float(v_samples.mean())

            calib_vectors[next_point_name] = (u_final, v_final)
            print(f"  Captured {next_point_name}: u={u_final:.4f}, v={v_final:.4f} ({args.calib_method}, n={len(u_samples)})")

        if len(calib_vectors) >= len(calib_point_names):
            cv2.imshow(calib_window, calib_screen)
            cv2.waitKey(500)
            break

    cv2.destroyWindow(calib_window)
    cv2.destroyWindow(preview_window)

    src = np.array([calib_vectors[name] for name in calib_point_names], dtype=np.float32)
    dst = screen_points

    if len(src) < 4:
        print(f"[ERROR] At least 4 calibration points needed, but only {len(src)} captured.")
        cap.release()
        cv2.destroyAllWindows()
        return

    method_flag = cv2.RANSAC if len(src) > 4 else 0
    M, _inliers = cv2.findHomography(src, dst, method=method_flag)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print("Running gaze heatmap. Press ESC to quit.\n")

    # Accumulate total heat without fading for the final image
    total_heat = np.zeros((H, W), dtype=np.float32)

    # ================================================================
    # MAIN HEATMAP LOOP
    # ================================================================
    heat = np.zeros((H, W), dtype=np.float32)

    smooth_alpha = float(args.smooth_alpha)
    smooth_alpha = max(0.0, min(1.0, smooth_alpha))

    smooth_mode = str(args.smooth_mode).lower()
    smooth_window = max(1, int(args.smooth_window))
    jump_px = float(args.jump_px)

    recent_points = []
    last_display_xy = None

    ref_center_x = float(W - 1) / 2.0
    ref_center_y = float(H - 1) / 2.0

    webcam_window = "Gaze Heatmap - Webcam"
    cv2.namedWindow(webcam_window, cv2.WINDOW_NORMAL)

    overlay_window = "Gaze Heatmap - Accumulating"
    if IS_WINDOWS and not args.no_overlay:
        setup_windows_overlay(overlay_window)
        try:
            cv2.setWindowProperty(overlay_window, cv2.WND_PROP_FULLSCREEN, 1)
        except Exception:
            pass
    else:
        cv2.namedWindow(overlay_window, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(overlay_window, cv2.WND_PROP_FULLSCREEN, 1)
        except Exception:
            pass

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_h, frame_w = frame.shape[0], frame.shape[1]
        landmarks = detector.detect(frame)
        vis = frame.copy()

        if landmarks is not None:
            vec, pts = extract_gaze_vector(landmarks, frame_w, frame_h, MP_API, iris_drop_ends=iris_drop_ends)
            draw_landmarks_on_frame(vis, pts)

            src_pt = np.array([[vec[0], vec[1]]], dtype=np.float32).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(src_pt, M)
            x, y = float(mapped[0, 0, 0]), float(mapped[0, 0, 1])

            if int(args.flip_x) == 1:
                x = (W - 1) - x
            if int(args.flip_y) == 1:
                y = (H - 1) - y

            x = clamp(x, 0.0, float(W - 1))
            y = clamp(y, 0.0, float(H - 1))

            # Head movement suppression: pull gaze toward screen center
            if head_suppression < 1.0:
                x = ref_center_x + head_suppression * (x - ref_center_x)
                y = ref_center_y + head_suppression * (y - ref_center_y)
                x = clamp(x, 0.0, float(W - 1))
                y = clamp(y, 0.0, float(H - 1))

            # Display smoothing
            sx, sy = x, y

            if last_display_xy is not None and jump_px > 0:
                px, py = last_display_xy
                dist = float(np.hypot(x - px, y - py))
                if dist > jump_px:
                    sx, sy = px, py
                else:
                    if smooth_mode == "ema":
                        prev_x, prev_y = last_display_xy
                        smoothed_x = smooth_alpha * x + (1.0 - smooth_alpha) * prev_x
                        smoothed_y = smooth_alpha * y + (1.0 - smooth_alpha) * prev_y
                        sx, sy = float(smoothed_x), float(smoothed_y)
                    else:
                        recent_points.append((x, y))
                        if len(recent_points) > smooth_window:
                            recent_points.pop(0)
                        pts_arr = np.array(recent_points, dtype=np.float32)
                        sx_med = float(np.median(pts_arr[:, 0]))
                        sy_med = float(np.median(pts_arr[:, 1]))
                        sx = smooth_alpha * sx_med + (1.0 - smooth_alpha) * px
                        sy = smooth_alpha * sy_med + (1.0 - smooth_alpha) * py
                    last_display_xy = (sx, sy)
            else:
                if smooth_mode == "ema":
                    sx, sy = x, y
                else:
                    recent_points.append((x, y))
                    if len(recent_points) > smooth_window:
                        recent_points.pop(0)
                    pts_arr = np.array(recent_points, dtype=np.float32)
                    sx = float(np.median(pts_arr[:, 0]))
                    sy = float(np.median(pts_arr[:, 1]))
                last_display_xy = (sx, sy)

            # Accumulate heatmap
            heat *= float(args.fade)
            blob = gaussian_blob((H, W), (x, y), float(args.sigma))
            if blob is not None:
                heat += blob.astype(np.float32)
                total_heat += blob.astype(np.float32)

            cam_px = (
                int((sx / (W - 1)) * (frame_w - 1)),
                int((sy / (H - 1)) * (frame_h - 1)),
            )
            cv2.circle(vis, cam_px, 8, (0, 0, 255), 2)
            cv2.putText(vis, f"Screen: ({int(sx)}, {int(sy)})",
                        (10, frame_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            heat *= float(args.fade)

        heat_thr = float(args.heat_bg_thresh)
        heat_masked = heat.copy()
        heat_masked[heat_masked < heat_thr] = 0.0

        heat_vis = cv2.normalize(heat_masked, None, 0, 255, cv2.NORM_MINMAX)
        heat_vis = heat_vis.astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_vis, cv2.COLORMAP_JET)

        if IS_WINDOWS and not args.no_overlay:
            # COLORMAP_JET maps 0 to dark blue, not black.
            # For proper color-key transparency, we need true black background.
            # Create a black canvas and only paste heatmap colors on actual heat areas.
            heat_color_u8 = heat_color.astype(np.uint8)
            
            # Mask where heat actually exists (after bg-thresholding)
            heat_exists = heat_masked > 0.0  # shape (H, W) bool
            
            # Start with pure black canvas
            overlay_canvas = np.zeros_like(heat_color_u8)
            
            # Copy heatmap colors only onto heat areas (everything else stays black)
            overlay_canvas[heat_exists] = heat_color_u8[heat_exists]
            
            # Ensure exact (0,0,0) for color-key transparency
            rgb_sum = overlay_canvas.astype(np.uint16).sum(axis=-1)
            near_black = rgb_sum < 8
            overlay_canvas[near_black] = (0, 0, 0)
            
            heat_color = overlay_canvas

        cv2.imshow(webcam_window, vis)
        cv2.imshow(overlay_window, heat_color)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if args.save_heatmap:
        print(f"\n[INFO] Saving final combined heatmap to {args.output}...")
        total_heat_thr = float(args.heat_bg_thresh)
        total_heat_masked = total_heat.copy()
        total_heat_masked[total_heat_masked < total_heat_thr] = 0.0
        
        if total_heat_masked.max() > 0:
            total_heat_vis = cv2.normalize(total_heat_masked, None, 0, 255, cv2.NORM_MINMAX)
            total_heat_vis = total_heat_vis.astype(np.uint8)
            total_heat_color = cv2.applyColorMap(total_heat_vis, cv2.COLORMAP_JET)
            
            # Mask out the background to be black instead of dark blue from COLORMAP_JET
            heat_exists = total_heat_masked > 0.0
            overlay_canvas = np.zeros_like(total_heat_color)
            overlay_canvas[heat_exists] = total_heat_color[heat_exists]
            
            cv2.imwrite(args.output, overlay_canvas)
            print(f"[INFO] Saved {args.output}")
        else:
            print("[INFO] No gaze data to save.")

    print("\nShutdown complete.")


if __name__ == "__main__":
    main()
