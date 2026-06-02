"""
mp_hands_helper.py — Unified MediaPipe hand detection wrapper.

Supports MediaPipe 0.10+ Tasks API (Python 3.11+).
Auto-downloads hand_landmarker.task model on first use.

Usage (from other scripts):
    from scripts.mp_hands_helper import HandDetector
    detector = HandDetector(mode="image")   # or "video"
    bbox = detector.get_bbox(bgr_frame, padding=0.20)
    detector.close()
"""

import os
import urllib.request
from pathlib import Path

# Model download URL (official MediaPipe CDN)
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_DIR  = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def ensure_model():
    """Download hand_landmarker.task if not present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"[INFO] Downloading hand landmark model → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")
    return str(MODEL_PATH)


class HandDetector:
    """
    Thin wrapper around MediaPipe HandLandmarker (Tasks API).

    mode="image"  → static image detection (for annotation scripts)
    mode="video"  → per-frame video detection (for demo / collect)
    """

    def __init__(self,
                 mode: str = "image",
                 max_hands: int = 1,
                 detection_confidence: float = 0.5,
                 presence_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_model()

        base_options = mp_python.BaseOptions(model_asset_path=model_path)

        if mode == "image":
            running_mode = mp_vision.RunningMode.IMAGE
        else:
            running_mode = mp_vision.RunningMode.VIDEO

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._mode = mode
        self._timestamp_ms = 0
        self._mp = mp

    def _detect(self, bgr_frame):
        """Run detection on a BGR numpy frame. Returns HandLandmarkerResult."""
        import numpy as np
        rgb = bgr_frame[:, :, ::-1].copy()   # BGR → RGB
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._mode == "image":
            return self._detector.detect(mp_image)
        else:
            self._timestamp_ms += 33   # ~30 fps
            return self._detector.detect_for_video(mp_image, self._timestamp_ms)

    def get_bbox(self, bgr_frame, padding: float = 0.20):
        """
        Returns (x1, y1, x2, y2) pixel bbox of the first detected hand,
        or None if no hand found.
        """
        result = self._detect(bgr_frame)
        if not result.hand_landmarks:
            return None
        h, w = bgr_frame.shape[:2]
        lm   = result.hand_landmarks[0]   # first hand
        xs   = [p.x for p in lm]
        ys   = [p.y for p in lm]
        pad_x = (max(xs) - min(xs)) * padding
        pad_y = (max(ys) - min(ys)) * padding
        x1 = max(0, int((min(xs) - pad_x) * w))
        y1 = max(0, int((min(ys) - pad_y) * h))
        x2 = min(w, int((max(xs) + pad_x) * w))
        y2 = min(h, int((max(ys) + pad_y) * h))
        return x1, y1, x2, y2

    def get_yolo_bbox(self, bgr_frame, padding: float = 0.20):
        """
        Returns (cx, cy, bw, bh) normalised YOLO bbox of first detected hand,
        or None if no hand found.
        """
        result = self._detect(bgr_frame)
        if not result.hand_landmarks:
            return None
        lm    = result.hand_landmarks[0]
        xs    = [p.x for p in lm]
        ys    = [p.y for p in lm]
        pad_x = (max(xs) - min(xs)) * padding
        pad_y = (max(ys) - min(ys)) * padding
        x1 = max(0.0, min(xs) - pad_x)
        y1 = max(0.0, min(ys) - pad_y)
        x2 = min(1.0, max(xs) + pad_x)
        y2 = min(1.0, max(ys) + pad_y)
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1

    def has_hand(self, bgr_frame) -> bool:
        """Returns True if a hand is detected in the frame."""
        result = self._detect(bgr_frame)
        return bool(result.hand_landmarks)

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
