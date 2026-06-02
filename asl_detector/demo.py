"""
demo.py — Live ASL sign detection using a trained YOLOv8 model and webcam.

Features:
    • Real-time bounding box around detected hand/sign
    • Predicted ASL class label + confidence score
    • Live FPS counter (rolling average)
    • MediaPipe pre-screening (skip YOLO on frames with no hand → lower CPU)
    • Keyboard shortcuts: S=screenshot, R=record, Q=quit

Usage:
    python demo.py
    python demo.py --weights models/asl_yolov8_best.pt --conf 0.45
    python demo.py --camera 1
    python demo.py --no_mediapipe   # disable pre-screening
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

np.random.seed(42)
PALETTE = {name: tuple(int(c) for c in np.random.randint(80, 230, 3))
           for name in CLASS_NAMES}


def parse_args():
    p = argparse.ArgumentParser(description="Live ASL sign detection demo")
    p.add_argument("--weights",  default=None)
    p.add_argument("--camera",   type=int,   default=0)
    p.add_argument("--conf",     type=float, default=0.40)
    p.add_argument("--iou",      type=float, default=0.45)
    p.add_argument("--imgsz",    type=int,   default=320)
    p.add_argument("--device",   default="cpu")
    p.add_argument("--fps_avg",  type=int,   default=30)
    p.add_argument("--no_mediapipe", action="store_true")
    return p.parse_args()


def find_weights():
    candidates = [f for f in os.listdir(MODELS_DIR) if f.endswith("_best.pt")]
    if candidates:
        return os.path.join(MODELS_DIR, sorted(candidates)[-1])
    runs_best = os.path.join(PROJECT_ROOT, "runs", "detect",
                             "asl_yolov8", "weights", "best.pt")
    return runs_best if os.path.exists(runs_best) else None


def draw_box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}  {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    by1 = max(0, y1 - th - baseline - 4)
    cv2.rectangle(frame, (x1, by1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_hud(frame, fps, predictions):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 230, 0), 2)
    if predictions:
        best  = max(predictions, key=lambda x: x[1])
        lbl, conf = best
        color = PALETTE.get(lbl, (0, 200, 255))
        cv2.putText(frame, f"{lbl}  {conf:.0%}", (w - 200, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame, "Q=quit  S=screenshot  R=record",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[ERROR] ultralytics not installed.")

    weights = args.weights or find_weights()
    if not weights or not os.path.exists(weights):
        sys.exit("[ERROR] No weights found. Train first or pass --weights <path>")

    print(f"[INFO] Weights   : {weights}")
    print(f"[INFO] Camera    : {args.camera}")
    print(f"[INFO] Conf thr  : {args.conf}")

    model  = YOLO(weights)
    device = args.device

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # Optional MediaPipe pre-screening
    mp_detector = None
    if not args.no_mediapipe:
        try:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
            from mp_hands_helper import HandDetector
            mp_detector = HandDetector(mode="video", detection_confidence=0.5)
            print("[INFO] MediaPipe pre-screening: enabled")
        except Exception as e:
            print(f"[WARN] MediaPipe unavailable ({e}) — running without pre-screening")

    fps_times: deque = deque(maxlen=args.fps_avg)
    recording    = False
    video_writer = None
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[INFO] Demo running. Press Q to quit.\n")

    while True:
        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        hand_present = True
        if mp_detector is not None:
            hand_present = mp_detector.has_hand(frame)

        predictions = []
        if hand_present:
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                    iou=args.iou, device=device, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = (r.names[cls_id] if r.names and cls_id in r.names
                              else CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES)
                              else str(cls_id))
                    color  = PALETTE.get(label, (0, 200, 255))
                    draw_box(frame, x1, y1, x2, y2, label, conf, color)
                    predictions.append((label, conf))

        fps_times.append(time.perf_counter() - t_start)
        fps = len(fps_times) / sum(fps_times)
        draw_hud(frame, fps, predictions)

        if recording and video_writer:
            video_writer.write(frame)

        cv2.imshow("ASL Live Demo — VANCO", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RESULTS_DIR, f"screenshot_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot: {path}")
        elif key == ord('r'):
            if not recording:
                ts    = time.strftime("%Y%m%d_%H%M%S")
                vpath = os.path.join(RESULTS_DIR, f"demo_{ts}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(vpath, fourcc, 20, (w, h))
                recording = True
                print(f"[INFO] Recording: {vpath}")
            else:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                print("[INFO] Recording stopped.")

    cap.release()
    if video_writer:
        video_writer.release()
    if mp_detector:
        mp_detector.close()
    cv2.destroyAllWindows()
    print("[INFO] Demo closed.")


if __name__ == "__main__":
    main()
