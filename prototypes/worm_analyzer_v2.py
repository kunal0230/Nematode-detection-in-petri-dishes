#!/usr/bin/env python3
"""
Worm Analyzer — Detection, Tracking & Analytics
=================================================
A professional web application for nematode/worm detection and tracking.

Launch:  python3 worm_analyzer.py
Opens:   http://localhost:7860

Features:
  - Image Analysis: Detect and count worms in images
  - Video Tracking: Track individual worms with IDs, velocity, paths
  - Batch Processing: Process multiple images at once
  - Analytics Dashboard: Charts, heatmaps, movement analysis
"""

import os
import sys
import cv2
import csv
import json
import time
import shutil
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_worm_yolov8m.pt")
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.5
PATCH_SIZE = 416

QUALITY_PRESETS = {
    "Fast":     {"overlap": 0.25, "skip_factor": 6, "desc": "Quick preview"},
    "Balanced": {"overlap": 0.40, "skip_factor": 3, "desc": "Recommended"},
    "Quality":  {"overlap": 0.50, "skip_factor": 1, "desc": "Maximum accuracy"},
}

# ============================================================================
# MODEL LOADING & HARDWARE BENCHMARK
# ============================================================================
print("Loading worm detection model...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Please place 'best_worm_yolov8m.pt' in the same directory as this script.")
    sys.exit(1)

model = YOLO(MODEL_PATH)
print(f"Model loaded: {model.names}")

import platform
import torch as _torch

def _detect_hardware():
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }
    if _torch.cuda.is_available():
        info["device"] = "CUDA"
        info["device_name"] = _torch.cuda.get_device_name(0)
        info["gpu_mem"] = f"{_torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        info["device"] = "MPS"
        info["device_name"] = "Apple Silicon GPU"
    else:
        info["device"] = "CPU"
        info["device_name"] = platform.processor() or "Unknown CPU"

    dummy = np.random.randint(0, 255, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    for _ in range(2):
        model(dummy, conf=0.5, imgsz=PATCH_SIZE, verbose=False)
    times = []
    for _ in range(3):
        t0 = time.time()
        model(dummy, conf=0.5, imgsz=PATCH_SIZE, verbose=False)
        times.append(time.time() - t0)
    info["patch_time_ms"] = np.median(times) * 1000
    return info

print("Benchmarking hardware...")
HW = _detect_hardware()
print(f"  Device: {HW['device']} ({HW['device_name']})")
print(f"  Speed: {HW['patch_time_ms']:.0f} ms per patch")

if HW["patch_time_ms"] < 80:
    DEFAULT_QUALITY = "Quality"
elif HW["patch_time_ms"] < 250:
    DEFAULT_QUALITY = "Balanced"
else:
    DEFAULT_QUALITY = "Fast"
print(f"  Default quality: {DEFAULT_QUALITY}")

import torch

# ============================================================================
# SLIDING WINDOW DETECTION (core engine)
# ============================================================================

def _get_patch_positions(img_w, img_h, patch_size, overlap):
    stride = int(patch_size * (1 - overlap))
    x_steps = list(range(0, max(1, img_w - patch_size + 1), stride))
    if len(x_steps) == 0 or x_steps[-1] + patch_size < img_w:
        x_steps.append(max(0, img_w - patch_size))
    y_steps = list(range(0, max(1, img_h - patch_size + 1), stride))
    if len(y_steps) == 0 or y_steps[-1] + patch_size < img_h:
        y_steps.append(max(0, img_h - patch_size))
    return x_steps, y_steps


def detect_worms(image, confidence=0.4, iou_threshold=0.5, patch_size=PATCH_SIZE, overlap=None):
    """
    Detect worms using sliding window inference with optional parallel processing.
    """
    if overlap is None:
        overlap = QUALITY_PRESETS[DEFAULT_QUALITY]["overlap"]

    h, w = image.shape[:2]

    # Small images: run directly
    if w <= patch_size * 1.5 and h <= patch_size * 1.5:
        results = model(image, conf=confidence, iou=iou_threshold, imgsz=PATCH_SIZE, verbose=False)
        result = results[0]
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    x_steps, y_steps = _get_patch_positions(w, h, patch_size, overlap)
    all_boxes = []
    all_scores = []

    # Build list of patches with positions
    patches = []
    positions = []
    for y_pos in y_steps:
        for x_pos in x_steps:
            patch = image[y_pos:y_pos+patch_size, x_pos:x_pos+patch_size]
            patches.append(patch)
            positions.append((x_pos, y_pos))

    # Batch inference: run all patches at once if GPU (much faster)
    # For CPU, batch of 1 is fine. For GPU, batch up to 8.
    batch_size = 8 if HW["device"] in ("CUDA", "MPS") else 1
    
    for batch_start in range(0, len(patches), batch_size):
        batch_patches = patches[batch_start:batch_start + batch_size]
        batch_positions = positions[batch_start:batch_start + batch_size]
        
        results = model(batch_patches, conf=confidence, iou=iou_threshold, imgsz=PATCH_SIZE, verbose=False)
        
        for result, (x_pos, y_pos) in zip(results, batch_positions):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                all_boxes.append([box[0] + x_pos, box[1] + y_pos, box[2] + x_pos, box[3] + y_pos])
                all_scores.append(score)

    if len(all_boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([])

    boxes_t = torch.tensor(np.array(all_boxes), dtype=torch.float32)
    scores_t = torch.tensor(np.array(all_scores), dtype=torch.float32)
    keep = torch.ops.torchvision.nms(boxes_t, scores_t, iou_threshold)

    return np.array(all_boxes)[keep.numpy()], np.array(all_scores)[keep.numpy()]


def _estimate_time(img_w, img_h, overlap, patch_size=PATCH_SIZE):
    if img_w <= patch_size * 1.5 and img_h <= patch_size * 1.5:
        return HW["patch_time_ms"] / 1000
    x_steps, y_steps = _get_patch_positions(img_w, img_h, patch_size, overlap)
    n_patches = len(x_steps) * len(y_steps)
    batch_size = 8 if HW["device"] in ("CUDA", "MPS") else 1
    n_batches = (n_patches + batch_size - 1) // batch_size
    return n_batches * HW["patch_time_ms"] / 1000


def _format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m}m"


# ============================================================================
# CENTROID TRACKER
# ============================================================================

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id = 1
        self.objects = {}
        self.boxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        if len(detections) == 0:
            to_remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    to_remove.append(oid)
            for oid in to_remove:
                del self.objects[oid]
                del self.boxes[oid]
                del self.disappeared[oid]
            return []

        new_centroids = np.array([
            ((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in detections
        ])

        if len(self.objects) == 0:
            results = []
            for i, (det, centroid) in enumerate(zip(detections, new_centroids)):
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = centroid
                self.boxes[tid] = det
                self.disappeared[tid] = 0
                results.append((tid, det))
            return results

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid] for oid in obj_ids])

        from scipy.spatial.distance import cdist
        dists = cdist(obj_centroids, new_centroids)

        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        results = []

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue
            oid = obj_ids[row]
            self.objects[oid] = new_centroids[col]
            self.boxes[oid] = detections[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)
            results.append((oid, detections[col]))

        for col in range(len(new_centroids)):
            if col not in used_cols:
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = new_centroids[col]
                self.boxes[tid] = detections[col]
                self.disappeared[tid] = 0
                results.append((tid, detections[col]))

        to_remove = []
        for row in range(len(obj_ids)):
            if row not in used_rows:
                oid = obj_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    to_remove.append(oid)
        for oid in to_remove:
            del self.objects[oid]
            del self.boxes[oid]
            del self.disappeared[oid]

        return results


# ============================================================================
# DRAWING UTILITIES
# ============================================================================

# Modern color palette - distinct, high-contrast colors
TRACK_COLORS = [
    (82, 196, 255),   # Sky blue
    (255, 128, 82),   # Coral
    (82, 255, 168),   # Mint
    (255, 220, 82),   # Yellow
    (200, 82, 255),   # Purple
    (82, 255, 230),   # Cyan
    (255, 160, 82),   # Orange
    (160, 82, 255),   # Violet
    (82, 255, 100),   # Green
    (255, 82, 180),   # Pink
    (82, 150, 255),   # Blue
    (255, 235, 82),   # Light yellow
]


def draw_detections(image, boxes, scores, track_ids=None, trails=None, show_conf=True, show_id=True):
    """Draw detection boxes with clean, modern styling."""
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Scale font/line thickness based on image size
    scale = max(0.4, min(1.2, w / 1000))
    line_w = max(1, int(2 * scale))
    font_scale = 0.45 * scale

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
        color = TRACK_COLORS[tid % len(TRACK_COLORS)] if tid is not None else (82, 255, 168)

        # Draw box with rounded feel (corner marks instead of full rect)
        corner_len = max(8, int((x2 - x1) * 0.2))
        # Top-left corner
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), color, line_w + 1)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), color, line_w + 1)
        # Top-right
        cv2.line(overlay, (x2 - corner_len, y1), (x2, y1), color, line_w + 1)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), color, line_w + 1)
        # Bottom-left
        cv2.line(overlay, (x1, y2 - corner_len), (x1, y2), color, line_w + 1)
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), color, line_w + 1)
        # Bottom-right
        cv2.line(overlay, (x2, y2 - corner_len), (x2, y2), color, line_w + 1)
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), color, line_w + 1)
        
        # Subtle fill
        sub = overlay[y1:y2, x1:x2]
        white = np.ones_like(sub, dtype=np.uint8) * np.array(color, dtype=np.uint8)
        overlay[y1:y2, x1:x2] = cv2.addWeighted(sub, 0.92, white, 0.08, 0)

        # Label
        parts = []
        if show_id and tid is not None:
            parts.append(f"#{tid}")
        if show_conf:
            parts.append(f"{score:.0%}")
        label = " ".join(parts)

        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            pad = 3
            label_y = max(th + pad * 2, y1)
            label_bg_y1 = label_y - th - pad * 2
            label_bg_y2 = label_y
            # Semi-transparent label background
            bg_sub = overlay[label_bg_y1:label_bg_y2, x1:x1 + tw + pad * 2]
            if bg_sub.shape[0] > 0 and bg_sub.shape[1] > 0:
                dark = np.zeros_like(bg_sub)
                overlay[label_bg_y1:label_bg_y2, x1:x1 + tw + pad * 2] = cv2.addWeighted(bg_sub, 0.3, dark, 0.7, 0)
            cv2.putText(overlay, label, (x1 + pad, label_y - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    # Draw tracking trails with fade
    if trails:
        trail_overlay = overlay.copy()
        for tid, points in trails.items():
            if len(points) < 2:
                continue
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            for j in range(1, len(points)):
                alpha = (j / len(points)) ** 1.5
                thickness = max(1, int(alpha * 3))
                pt1 = tuple(map(int, points[j-1]))
                pt2 = tuple(map(int, points[j]))
                # Draw with alpha fade
                cv2.line(trail_overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
        overlay = cv2.addWeighted(overlay, 0.7, trail_overlay, 0.3, 0)

    return overlay


# ============================================================================
# TAB 1: IMAGE ANALYSIS
# ============================================================================

def analyze_image(image, confidence, iou_threshold, quality):
    if image is None:
        return None, "No image uploaded.", None

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
    overlap = preset["overlap"]

    start = time.time()
    boxes, scores = detect_worms(image, confidence=confidence, iou_threshold=iou_threshold, overlap=overlap)
    elapsed = time.time() - start

    n_det = len(boxes)
    annotated = draw_detections(image, boxes, scores, show_conf=True, show_id=False)
    h, w = annotated.shape[:2]

    # Clean HUD overlay
    hud_h, hud_w = 90, 320
    hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
    cv2.rectangle(annotated, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0), -1)
    sub = annotated[10:10+hud_h, 10:10+hud_w]
    annotated[10:10+hud_h, 10:10+hud_w] = cv2.addWeighted(sub, 0.2, hud, 0.8, 0)
    
    cv2.putText(annotated, f"{n_det} Worms Detected", (22, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (82, 255, 168), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"{elapsed:.1f}s  {quality}  {HW['device']}", (22, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    
    x_steps, y_steps = _get_patch_positions(w, h, PATCH_SIZE, overlap)
    n_patches = len(x_steps) * len(y_steps) if (w > PATCH_SIZE * 1.5 or h > PATCH_SIZE * 1.5) else 1

    stats_lines = [
        f"## {n_det} Worms Detected",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Processing time | {elapsed:.1f}s ({HW['device']}) |",
        f"| Image size | {w} × {h} px |",
        f"| Patches scanned | {n_patches} ({PATCH_SIZE}×{PATCH_SIZE}, {int(overlap*100)}% overlap) |",
        f"| Quality preset | {quality} |",
        f"| Confidence threshold | {confidence} |",
    ]
    if n_det > 0:
        avg_conf = np.mean(scores)
        widths_px = boxes[:, 2] - boxes[:, 0]
        heights_px = boxes[:, 3] - boxes[:, 1]
        areas = widths_px * heights_px
        stats_lines.extend([
            f"| Avg confidence | {avg_conf:.1%} |",
            f"| Avg worm size | {np.mean(widths_px):.0f} × {np.mean(heights_px):.0f} px |",
            f"| Size range | {np.min(areas):.0f} – {np.max(areas):.0f} px² |",
        ])

    csv_path = None
    if n_det > 0:
        csv_path = os.path.join(tempfile.gettempdir(), "worm_detections.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["worm_id", "x1", "y1", "x2", "y2", "confidence", "width_px", "height_px", "area_px2"])
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box
                writer.writerow([
                    i + 1, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{score:.4f}", f"{x2-x1:.1f}", f"{y2-y1:.1f}", f"{(x2-x1)*(y2-y1):.0f}"
                ])

    return annotated, "\n".join(stats_lines), csv_path


# ============================================================================
# TAB 2: VIDEO TRACKING
# ============================================================================

def track_video(video_path, confidence, iou_threshold, quality, trail_length,
                crop_x1=0, crop_y1=0, crop_x2=0, crop_y2=0,
                trim_start=0, trim_end=0, progress=gr.Progress()):
    if video_path is None:
        return None, "No video uploaded.", None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video file.", None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    crop_x1, crop_y1, crop_x2, crop_y2 = int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)
    use_crop = not (crop_x1 == 0 and crop_y1 == 0 and crop_x2 == 0 and crop_y2 == 0)
    if use_crop:
        crop_x1 = max(0, min(crop_x1, orig_w - 1))
        crop_y1 = max(0, min(crop_y1, orig_h - 1))
        crop_x2 = max(crop_x1 + 10, min(crop_x2, orig_w))
        crop_y2 = max(crop_y1 + 10, min(crop_y2, orig_h))
    else:
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, orig_w, orig_h

    w = crop_x2 - crop_x1
    h = crop_y2 - crop_y1

    trim_start = max(0, float(trim_start))
    trim_end = float(trim_end)
    if trim_end <= trim_start:
        trim_end = duration
    start_frame = int(trim_start * fps)
    end_frame = min(int(trim_end * fps), total_frames)

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
    video_overlap = preset["overlap"]
    frame_skip = max(1, preset["skip_factor"])

    trimmed_frames = end_frame - start_frame
    frames_to_process = trimmed_frames // frame_skip
    per_frame_est = _estimate_time(w, h, video_overlap)
    total_est = frames_to_process * per_frame_est

    out_path = os.path.join(tempfile.gettempdir(), "tracked_worms.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    tracker = CentroidTracker(max_disappeared=int(fps * 2), max_distance=80)

    frame_counts = []
    all_tracks = {}
    trails = defaultdict(list)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = 0
    processed_count = 0
    last_annotated = None
    proc_start = time.time()
    eta = total_est

    progress(0, desc=f"Starting — ETA {_format_eta(total_est)} ({quality}, {HW['device']})")

    while frame_num < trimmed_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if use_crop:
            frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if frame_num % frame_skip == 0:
            boxes, scores = detect_worms(
                frame, confidence=confidence, iou_threshold=iou_threshold,
                overlap=video_overlap
            )

            tracked = tracker.update(boxes)

            track_ids_list = []
            tracked_boxes = []
            for tid, box in tracked:
                track_ids_list.append(tid)
                tracked_boxes.append(box)
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                if tid not in all_tracks:
                    all_tracks[tid] = {"first_seen": frame_num, "positions": []}
                all_tracks[tid]["last_seen"] = frame_num
                all_tracks[tid]["positions"].append((cx, cy, frame_num))
                trails[tid].append((cx, cy))
                if len(trails[tid]) > trail_length:
                    trails[tid] = trails[tid][-trail_length:]

            tracked_boxes = np.array(tracked_boxes) if tracked_boxes else np.array([]).reshape(0, 4)
            track_ids_arr = np.array(track_ids_list) if track_ids_list else np.array([])
            tracked_scores = scores[:len(tracked_boxes)] if len(scores) >= len(tracked_boxes) else np.ones(len(tracked_boxes))

            frame_counts.append(len(tracked_boxes))

            annotated = draw_detections(
                frame, tracked_boxes, tracked_scores,
                track_ids=track_ids_arr if len(track_ids_arr) > 0 else None,
                trails=trails if len(trails) > 0 else None,
                show_conf=False, show_id=True
            )

            # HUD
            hud_h_size = 100
            hud = np.zeros((hud_h_size, 370, 3), dtype=np.uint8)
            cv2.rectangle(annotated, (10, 10), (380, 10 + hud_h_size), (0, 0, 0), -1)
            sub = annotated[10:10+hud_h_size, 10:380]
            annotated[10:10+hud_h_size, 10:380] = cv2.addWeighted(sub, 0.2, hud, 0.8, 0)

            cv2.putText(annotated, f"{len(tracked_boxes)} Worms  |  {len(all_tracks)} Unique IDs", (22, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (82, 255, 168), 2, cv2.LINE_AA)
            time_sec = frame_num / fps
            cv2.putText(annotated, f"t={time_sec:.1f}s  frame={frame_num}/{total_frames}", (22, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{quality}  {HW['device']}", (22, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

            last_annotated = annotated
            processed_count += 1

            actual_per_frame = (time.time() - proc_start) / processed_count
            remaining_frames = frames_to_process - processed_count
            eta = remaining_frames * actual_per_frame
        else:
            annotated = last_annotated if last_annotated is not None else frame

        out.write(annotated)
        frame_num += 1

        if frame_num % 10 == 0:
            pct = frame_num / trimmed_frames
            progress(pct, desc=f"Frame {frame_num}/{trimmed_frames} — {len(all_tracks)} tracks — ETA {_format_eta(eta) if processed_count > 0 else '...'}")

    cap.release()
    out.release()

    duration_out = trimmed_frames / fps

    tracking_csv_path = os.path.join(tempfile.gettempdir(), "tracking_data.csv")
    with open(tracking_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "first_frame", "last_frame", "duration_sec",
                         "total_distance_px", "avg_velocity_px_per_sec",
                         "start_x", "start_y", "end_x", "end_y",
                         "displacement_px", "n_detections"])
        for tid, data in sorted(all_tracks.items()):
            positions = data["positions"]
            first = data["first_seen"]
            last = data["last_seen"]
            dur = (last - first) / fps if last != first else 0
            total_dist = sum(
                np.sqrt((positions[i][0]-positions[i-1][0])**2 + (positions[i][1]-positions[i-1][1])**2)
                for i in range(1, len(positions))
            )
            avg_vel = total_dist / dur if dur > 0 else 0
            start_pos = positions[0]
            end_pos = positions[-1]
            displacement = np.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)
            writer.writerow([tid, first, last, f"{dur:.2f}",
                             f"{total_dist:.1f}", f"{avg_vel:.2f}",
                             f"{start_pos[0]:.1f}", f"{start_pos[1]:.1f}",
                             f"{end_pos[0]:.1f}", f"{end_pos[1]:.1f}",
                             f"{displacement:.1f}", len(positions)])

    total_proc_time = time.time() - proc_start

    stats_lines = [
        f"## Tracking Complete",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Duration | {duration_out:.1f}s ({trimmed_frames} frames @ {fps:.0f} fps) |",
        f"| Resolution | {w} × {h}" + (f" (cropped)" if use_crop else "") + " |",
        f"| Mode | {quality} on {HW['device']}, every {frame_skip} frames |",
        f"| Processing time | {_format_eta(total_proc_time)} |",
        f"| **Unique worms tracked** | **{len(all_tracks)}** |",
        f"| Avg worms/frame | {np.mean(frame_counts):.1f}" if frame_counts else "| Detected | None |",
        f"| Peak worms | {max(frame_counts) if frame_counts else 0} |",
    ]

    if all_tracks:
        velocities, distances = [], []
        for tid, data in all_tracks.items():
            positions = data["positions"]
            if len(positions) < 2:
                continue
            total_dist = sum(
                np.sqrt((positions[i][0]-positions[i-1][0])**2 + (positions[i][1]-positions[i-1][1])**2)
                for i in range(1, len(positions))
            )
            dur = (data["last_seen"] - data["first_seen"]) / fps
            if dur > 0:
                velocities.append(total_dist / dur)
            distances.append(total_dist)

        if velocities:
            stats_lines.extend([
                f"",
                f"**Movement**",
                f"| Avg velocity | {np.mean(velocities):.1f} px/s |",
                f"| Max velocity | {np.max(velocities):.1f} px/s |",
                f"| Avg distance | {np.mean(distances):.0f} px |",
                f"| Max distance | {np.max(distances):.0f} px |",
            ])

    analytics_img = generate_analytics(all_tracks, frame_counts, fps, w, h)
    return out_path, "\n".join(stats_lines), tracking_csv_path, analytics_img


def generate_analytics(all_tracks, frame_counts, fps, img_w, img_h):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib import rcParams

        # Clean scientific style
        BG = "#0f1117"
        PANEL = "#1a1d27"
        ACCENT = "#52c4ff"
        ACCENT2 = "#52ffa8"
        TEXT = "#e2e8f0"
        GRID = "#2a2d3a"

        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                                left=0.07, right=0.97, top=0.90, bottom=0.10)

        ax1 = fig.add_subplot(gs[0, :2])  # Wide: count over time
        ax2 = fig.add_subplot(gs[0, 2])   # Velocity distribution
        ax3 = fig.add_subplot(gs[1, :2])  # Paths
        ax4 = fig.add_subplot(gs[1, 2])   # Heatmap

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.title.set_color(TEXT)
            ax.title.set_fontsize(10)
            ax.title.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_color(GRID)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

        # 1. Worm count over time
        times = [i * (fps if fps else 30) / max(1, fps) for i in range(len(frame_counts))]
        times_sec = [i / max(1, fps) for i in range(len(frame_counts))]
        if frame_counts:
            ax1.plot(times_sec, frame_counts, color=ACCENT, linewidth=1.5, alpha=0.9)
            ax1.fill_between(times_sec, frame_counts, alpha=0.15, color=ACCENT)
            # Rolling avg
            if len(frame_counts) > 10:
                window = max(3, len(frame_counts) // 20)
                rolling = np.convolve(frame_counts, np.ones(window)/window, mode='valid')
                rolling_t = times_sec[window//2:window//2 + len(rolling)]
                ax1.plot(rolling_t, rolling, color=ACCENT2, linewidth=2, linestyle="--", alpha=0.8, label="Rolling avg")
                ax1.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
        ax1.set_xlabel("Time (s)", fontsize=8)
        ax1.set_ylabel("Worm Count", fontsize=8)
        ax1.set_title("Worm Count Over Time")

        # 2. Velocity distribution
        velocities = []
        for tid, data in all_tracks.items():
            positions = data["positions"]
            if len(positions) < 2:
                continue
            total_dist = sum(
                np.sqrt((positions[i][0]-positions[i-1][0])**2 + (positions[i][1]-positions[i-1][1])**2)
                for i in range(1, len(positions))
            )
            dur = (data["last_seen"] - data["first_seen"]) / fps
            if dur > 0:
                velocities.append(total_dist / dur)

        if velocities:
            ax2.hist(velocities, bins=15, color=ACCENT, edgecolor=BG, alpha=0.85)
            ax2.axvline(np.mean(velocities), color=ACCENT2, linewidth=1.5, linestyle="--", label=f"Mean {np.mean(velocities):.0f}")
            ax2.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=7)
        ax2.set_xlabel("Velocity (px/s)", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.set_title("Velocity Distribution")

        # 3. Movement paths
        colors_cmap = plt.cm.plasma(np.linspace(0.1, 0.9, max(1, len(all_tracks))))
        for idx, (tid, data) in enumerate(all_tracks.items()):
            positions = data["positions"]
            if len(positions) < 2:
                continue
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            ax3.plot(xs, ys, color=colors_cmap[idx % len(colors_cmap)], linewidth=0.8, alpha=0.7)
            ax3.scatter(xs[0], ys[0], color=ACCENT2, s=15, zorder=5, alpha=0.8)
            ax3.scatter(xs[-1], ys[-1], color="#ff5252", s=15, zorder=5, marker="x", linewidths=1.5)
        ax3.set_xlim(0, img_w)
        ax3.set_ylim(img_h, 0)
        ax3.set_xlabel("X (px)", fontsize=8)
        ax3.set_ylabel("Y (px)", fontsize=8)
        ax3.set_title("Movement Paths  (● start  × end)")
        ax3.set_aspect("equal", adjustable="datalim")

        # 4. Activity heatmap
        scale = 8
        heatmap = np.zeros((img_h // scale + 1, img_w // scale + 1), dtype=np.float32)
        for tid, data in all_tracks.items():
            for x, y, _ in data["positions"]:
                hx = min(int(x / scale), heatmap.shape[1] - 1)
                hy = min(int(y / scale), heatmap.shape[0] - 1)
                heatmap[hy, hx] += 1

        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=3)
        im = ax4.imshow(heatmap, cmap="inferno", aspect="auto", origin="upper")
        plt.colorbar(im, ax=ax4, label="Activity", shrink=0.85).ax.tick_params(labelcolor=TEXT, labelsize=7)
        ax4.set_title("Activity Heatmap")
        ax4.set_xlabel("X", fontsize=8)
        ax4.set_ylabel("Y", fontsize=8)

        fig.suptitle("Worm Tracking Analytics", fontsize=14, color=TEXT, fontweight="bold", y=0.97)

        out_path = os.path.join(tempfile.gettempdir(), "analytics.png")
        plt.savefig(out_path, dpi=130, facecolor=BG, bbox_inches="tight")
        plt.close()

        img = cv2.imread(out_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError as e:
        print(f"Analytics requires matplotlib and scipy: {e}")
        return None


# ============================================================================
# TAB 3: BATCH PROCESSING
# ============================================================================

def process_batch(files, confidence, iou_threshold, quality, progress=gr.Progress()):
    if not files:
        return None, "No images uploaded.", None

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
    overlap = preset["overlap"]

    results_dir = os.path.join(tempfile.gettempdir(), "worm_batch_results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    all_stats = []
    gallery_images = []

    for i, file_path in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {i+1}/{len(files)}: {os.path.basename(file_path)}")

        img = cv2.imread(file_path)
        if img is None:
            continue

        fname = os.path.basename(file_path)
        boxes, scores = detect_worms(img, confidence=confidence, iou_threshold=iou_threshold, overlap=overlap)
        annotated = draw_detections(img, boxes, scores, show_conf=True, show_id=False)

        # Clean count overlay
        count_label = f"{len(boxes)} worms"
        cv2.rectangle(annotated, (10, 10), (200, 52), (0, 0, 0), -1)
        sub = annotated[10:52, 10:200]
        dark = np.zeros_like(sub)
        annotated[10:52, 10:200] = cv2.addWeighted(sub, 0.2, dark, 0.8, 0)
        cv2.putText(annotated, count_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (82, 255, 168), 2, cv2.LINE_AA)

        out_path = os.path.join(results_dir, f"detected_{fname}")
        if not out_path.lower().endswith((".jpg", ".jpeg", ".png")):
            out_path += ".jpg"
        cv2.imwrite(out_path, annotated)

        gallery_images.append((cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), f"{fname} — {len(boxes)} worms"))

        avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0
        all_stats.append({
            "filename": fname,
            "worm_count": len(boxes),
            "avg_confidence": f"{avg_conf:.2%}",
        })

    csv_path = os.path.join(results_dir, "batch_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "worm_count", "avg_confidence"])
        writer.writeheader()
        writer.writerows(all_stats)

    zip_path = os.path.join(tempfile.gettempdir(), "worm_batch_results")
    shutil.make_archive(zip_path, "zip", results_dir)

    total = sum(s["worm_count"] for s in all_stats)
    avg = total / len(all_stats) if all_stats else 0

    summary = [
        f"## Batch Complete: {len(all_stats)} images",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Images processed | {len(all_stats)} |",
        f"| Total worms | {total} |",
        f"| Average per image | {avg:.1f} |",
        f"",
        f"### Per-Image Results",
        f"| Image | Worms | Confidence |",
        f"|-------|-------|------------|",
    ]
    for s in all_stats:
        summary.append(f"| {s['filename']} | {s['worm_count']} | {s['avg_confidence']} |")

    return gallery_images, "\n".join(summary), zip_path + ".zip"


# ============================================================================
# TAB 4: FRAME EXPLORER
# ============================================================================

def analyze_frame_by_frame(video_path, confidence, frame_number,
                           crop_x1=0, crop_y1=0, crop_x2=0, crop_y2=0):
    if video_path is None:
        return None, "No video loaded."

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = min(max(0, int(frame_number)), total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Cannot read frame."

    crop_x1, crop_y1, crop_x2, crop_y2 = int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)
    use_crop = not (crop_x1 == 0 and crop_y1 == 0 and crop_x2 == 0 and crop_y2 == 0)
    if use_crop:
        crop_x1 = max(0, min(crop_x1, orig_w - 1))
        crop_y1 = max(0, min(crop_y1, orig_h - 1))
        crop_x2 = max(crop_x1 + 10, min(crop_x2, orig_w))
        crop_y2 = max(crop_y1 + 10, min(crop_y2, orig_h))
        frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    boxes, scores = detect_worms(frame, confidence=confidence, iou_threshold=0.5)
    annotated = draw_detections(frame, boxes, scores, show_conf=True, show_id=False)
    time_sec = frame_number / fps

    hud = np.zeros((80, 310, 3), dtype=np.uint8)
    cv2.rectangle(annotated, (10, 10), (320, 90), (0, 0, 0), -1)
    sub = annotated[10:90, 10:320]
    annotated[10:90, 10:320] = cv2.addWeighted(sub, 0.2, hud, 0.8, 0)
    cv2.putText(annotated, f"{len(boxes)} Worms", (22, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (82, 255, 168), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Frame {frame_number}  t={time_sec:.2f}s", (22, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    stats = f"**Frame {frame_number}** ({time_sec:.2f}s) — **{len(boxes)} worms** detected | Avg conf: {np.mean(scores):.1%}" if len(scores) > 0 else f"**Frame {frame_number}** ({time_sec:.2f}s) — **0 worms** detected"
    return annotated, stats


# ============================================================================
# VIDEO TOOLS
# ============================================================================

def auto_detect_dish(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for better contrast in poor illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    h, w = frame.shape[:2]
    min_r = min(h, w) // 6
    max_r = min(h, w) // 2

    for dp in [1.2, 1.5, 2.0]:
        for param1 in [80, 50, 30]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min(h, w) // 2,
                param1=param1, param2=50,
                minRadius=min_r, maxRadius=max_r
            )
            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                best = max(circles, key=lambda c: c[2])
                return int(best[0]), int(best[1]), int(best[2])

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        if radius > min_r:
            return int(cx), int(cy), int(radius)

    return None


def _get_crop_from_dish(cx, cy, r, padding, w, h):
    if cx == 0 and cy == 0 and r == 0:
        return 0, 0, w, h
    pad = int(padding)
    x1 = max(0, cx - r - pad)
    y1 = max(0, cy - r - pad)
    x2 = min(w, cx + r + pad)
    y2 = min(h, cy + r + pad)
    return x1, y1, x2, y2


def crop_and_trim_video(video_path, x1, y1, x2, y2, start_sec, end_sec, progress=gr.Progress()):
    if video_path is None:
        return None, "No video uploaded.", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video.", None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(x1 + 10, min(x2, orig_w))
    y2 = max(y1 + 10, min(y2, orig_h))
    start_sec = max(0, float(start_sec))
    end_sec = min(duration, float(end_sec))
    if end_sec <= start_sec:
        end_sec = duration

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    out_frames = end_frame - start_frame
    crop_w, crop_h = x2 - x1, y2 - y1

    progress(0, desc=f"Processing {out_frames} frames...")
    out_path = os.path.join(tempfile.gettempdir(), "cropped_trimmed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    proc_start = time.time()
    written = 0
    for i in range(out_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame[y1:y2, x1:x2])
        written += 1
        if written % 100 == 0:
            elapsed = time.time() - proc_start
            rate = written / elapsed if elapsed > 0 else 1
            eta = (out_frames - written) / rate
            progress(written / out_frames, desc=f"Frame {written}/{out_frames} ({rate:.0f} fps) — ETA {_format_eta(eta)}")

    cap.release()
    out.release()

    total_time = time.time() - proc_start
    rate = written / total_time if total_time > 0 else 0
    speedup = orig_w * orig_h / (crop_w * crop_h) if crop_w * crop_h > 0 else 1

    stats = (
        f"## Video Processed\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Output size | {crop_w} × {crop_h} px |\n"
        f"| Duration | {end_sec - start_sec:.1f}s ({written} frames) |\n"
        f"| Processing | {total_time:.1f}s ({rate:.0f} frames/s) |\n"
        f"| Size reduction | {100*(1-crop_w*crop_h/(orig_w*orig_h)):.0f}% smaller |\n"
        f"| Detection speedup | ~{speedup:.1f}× faster |\n"
    )
    return out_path, stats, out_path


# ============================================================================
# GRADIO UI — Professional Dark Scientific Theme
# ============================================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #0b0e14;
    --surface: #12151e;
    --surface2: #1a1f2e;
    --surface3: #232940;
    --border: #2a2f45;
    --border-active: #3d4a6e;
    --accent: #52c4ff;
    --accent2: #52ffa8;
    --accent3: #ff7b52;
    --text: #dde3f0;
    --text-muted: #7a8499;
    --text-dim: #4a5568;
    --danger: #ff5252;
    --radius: 8px;
    --radius-sm: 5px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ---- App header ---- */
.app-header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.app-header h1 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin: 0 !important;
    letter-spacing: -0.3px;
}
.app-header .badge {
    background: var(--accent);
    color: #000;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 99px;
    letter-spacing: 0.5px;
}
.hw-badge {
    margin-left: auto;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
}

/* ---- Tabs ---- */
.tabs {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}
.tab-nav {
    background: transparent !important;
    padding: 0 20px !important;
    gap: 0 !important;
}
.tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 12px 18px !important;
    margin: 0 !important;
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
}
.tab-nav button:hover {
    color: var(--text) !important;
    background: rgba(82, 196, 255, 0.05) !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ---- Layout panels ---- */
.panel-left {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}
.panel-right {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}

/* ---- Generic block overrides ---- */
.block, .form {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.block.padded { padding: 16px !important; }

/* ---- Labels ---- */
label span, .label-wrap span {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- Sliders ---- */
input[type="range"] {
    accent-color: var(--accent) !important;
}
.slider-container .head { color: var(--text) !important; }

/* ---- Number inputs ---- */
input[type="number"], input[type="text"], textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}
input[type="number"]:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(82, 196, 255, 0.15) !important;
}

/* ---- Checkboxes ---- */
input[type="checkbox"] {
    accent-color: var(--accent) !important;
}

/* ---- Radio buttons ---- */
.wrap { color: var(--text) !important; }
.wrap input[type="radio"] { accent-color: var(--accent) !important; }

/* ---- Buttons ---- */
button.primary, .btn-primary {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    letter-spacing: 0.2px !important;
    transition: all 0.15s !important;
}
button.primary:hover {
    background: #7dd6ff !important;
    transform: translateY(-1px) !important;
}
button.secondary {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
button.secondary:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ---- Accordions ---- */
.accordion { 
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.accordion .label-wrap { 
    padding: 10px 14px !important;
    font-size: 12px !important;
}
.accordion-content { 
    background: var(--surface2) !important;
    padding: 12px 14px !important;
}

/* ---- Markdown ---- */
.prose, .md {
    color: var(--text) !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.prose h2, .md h2 { 
    color: var(--text) !important; 
    font-size: 15px !important;
    font-weight: 600 !important;
    margin: 0 0 10px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid var(--border) !important;
}
.prose h3, .md h3 { 
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    margin: 16px 0 8px !important;
}
.prose table, .md table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 12px !important;
    font-family: 'DM Mono', monospace !important;
    margin-top: 8px !important;
}
.prose th, .md th {
    background: var(--surface3) !important;
    color: var(--text-muted) !important;
    padding: 6px 10px !important;
    text-align: left !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.prose td, .md td {
    padding: 5px 10px !important;
    border-bottom: 1px solid var(--border) !important;
    color: var(--text) !important;
}
.prose tr:last-child td, .md tr:last-child td { border-bottom: none !important; }
.prose strong, .md strong { color: var(--accent2) !important; }

/* ---- Images & Video ---- */
.image-container img, .video-container video {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ---- Upload areas ---- */
.upload-button, .upload-zone {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-muted) !important;
    transition: all 0.2s !important;
}
.upload-zone:hover {
    border-color: var(--accent) !important;
    background: rgba(82, 196, 255, 0.05) !important;
}

/* ---- File component ---- */
.file-component {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ---- Gallery ---- */
.grid-container {
    background: var(--surface) !important;
}
.gallery-item {
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

/* ---- Section headers ---- */
.section-label {
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--text-dim) !important;
    padding: 8px 0 4px !important;
    border-top: 1px solid var(--border) !important;
    margin-top: 12px !important;
}

/* ---- Status/info chips ---- */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--surface3);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
}

/* ---- Scrollbars ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-active); }

/* ---- Footer ---- */
footer { display: none !important; }
.footer-note {
    text-align: center;
    padding: 16px;
    color: var(--text-dim);
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    border-top: 1px solid var(--border);
    margin-top: 20px;
}
"""

app_theme = gr.themes.Base(
    font=gr.themes.GoogleFont("DM Sans"),
    font_mono=gr.themes.GoogleFont("DM Mono"),
)

device_str = HW.get("device_name", HW["device"])[:30]

with gr.Blocks(title="Worm Analyzer", theme=app_theme, css=custom_css) as app:

    # ---- Header ----
    gr.HTML(f"""
    <div class="app-header">
        <div style="width:32px; height:32px; border-radius:8px; background:linear-gradient(135deg,#52c4ff,#52ffa8); display:flex; align-items:center; justify-content:center;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2.5" stroke-linecap="round">
                <path d="M4 12c0-3 2-5 4-5s3 2 5 2 3-2 5-2 4 2 4 5-2 5-4 5-3-2-5-2-3 2-5 2-4-2-4-5z"/>
            </svg>
        </div>
        <h1>Worm Analyzer</h1>
        <span class="badge">YOLOv8m</span>
        <div class="hw-badge">{HW['device']} · {device_str} · {HW['patch_time_ms']:.0f}ms/patch · {DEFAULT_QUALITY} default</div>
    </div>
    """)

    with gr.Tabs(elem_classes=["main-tabs"]):

        # ================================================================
        # TAB 1: IMAGE ANALYSIS
        # ================================================================
        with gr.Tab("Image Analysis"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    gr.HTML('<div class="section-label">Input</div>')
                    img_input = gr.Image(type="numpy", label="Upload Image", height=350)
                    
                    gr.HTML('<div class="section-label">Detection Settings</div>')
                    img_quality = gr.Radio(
                        choices=["Fast", "Balanced", "Quality"],
                        value=DEFAULT_QUALITY,
                        label="Quality",
                        info="Fast=25% overlap · Balanced=40% · Quality=50%"
                    )
                    with gr.Row():
                        img_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05,
                                             label="Confidence", info="Detection threshold")
                        img_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05,
                                            label="NMS IoU", info="Overlap threshold")
                    img_btn = gr.Button("Analyze Image", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-label">Result</div>')
                    img_output = gr.Image(label="Detected Worms", height=430)
                    with gr.Row():
                        with gr.Column(scale=3):
                            img_stats = gr.Markdown("*Upload an image and click Analyze.*")
                        with gr.Column(scale=1, min_width=160):
                            img_csv = gr.File(label="Export CSV")

            img_btn.click(
                analyze_image,
                inputs=[img_input, img_conf, img_iou, img_quality],
                outputs=[img_output, img_stats, img_csv],
            )

        # ================================================================
        # TAB 2: VIDEO TRACKING
        # ================================================================
        with gr.Tab("Video Tracking"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300):
                    gr.HTML('<div class="section-label">Input</div>')
                    vid_input = gr.Video(label="Upload Video", height=240)
                    
                    gr.HTML('<div class="section-label">Crop & Region</div>')
                    with gr.Row():
                        vid_auto_crop = gr.Checkbox(label="Auto-detect dish", value=True)
                        vid_pad = gr.Slider(-50, 150, value=50, step=5, label="Padding (px)")
                    vid_crop_preview = gr.Image(label="Crop Preview", height=180)

                    vid_cx1 = gr.Number(value=0, visible=False, precision=0)
                    vid_cy1 = gr.Number(value=0, visible=False, precision=0)
                    vid_cx2 = gr.Number(value=0, visible=False, precision=0)
                    vid_cy2 = gr.Number(value=0, visible=False, precision=0)

                    gr.HTML('<div class="section-label">Detection</div>')
                    vid_quality = gr.Radio(["Fast", "Balanced", "Quality"], value=DEFAULT_QUALITY, label="Quality")
                    with gr.Row():
                        vid_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence")
                        vid_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05, label="NMS IoU")
                    
                    with gr.Accordion("Advanced", open=False):
                        vid_trail = gr.Slider(10, 200, value=60, step=10, label="Trail Length")
                        gr.Markdown("**Trim (seconds)**")
                        with gr.Row():
                            vid_trim_s = gr.Number(value=0, label="Start", precision=1)
                            vid_trim_e = gr.Number(value=0, label="End (0=full)", precision=1)

                    vid_btn = gr.Button("Start Tracking", variant="primary", size="lg")

                with gr.Column(scale=3):
                    gr.HTML('<div class="section-label">Output</div>')
                    vid_output = gr.Video(label="Tracked Video", height=480)
                    with gr.Row():
                        with gr.Column(scale=3):
                            vid_analytics = gr.Image(label="Analytics", height=300)
                        with gr.Column(scale=1, min_width=200):
                            vid_stats = gr.Markdown("*Run tracking to see stats.*")
                            vid_csv = gr.File(label="Export CSV")

            vid_dish_data = gr.State(value=None)
            vid_frame_state = gr.State(value=None)

            def _on_video_upload(video_path):
                if video_path is None:
                    return None, None, 0, 0, 0, 0, None, 0, 0
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None, None, 0, 0, 0, 0, None, 0, 0
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur = total / fps
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None, None, 0, 0, 0, 0, None, 0, 0
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dish = auto_detect_dish(frame)
                dish_data = {"cx": 0, "cy": 0, "r": 0, "w": w, "h": h}
                if dish:
                    cx, cy, r = dish
                    dish_data = {"cx": cx, "cy": cy, "r": r, "w": w, "h": h}
                x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], 50, w, h)
                preview = frame_rgb.copy()
                if dish:
                    cv2.circle(preview, (dish[0], dish[1]), dish[2], (82, 255, 168), 2)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 220, 82), 3)
                return dish_data, frame_rgb, x1, y1, x2, y2, preview, 0, round(dur, 1)

            def _update_crop_preview(dish_data, frame, auto_crop, padding):
                if dish_data is None or frame is None:
                    return 0, 0, 0, 0, None
                w, h = dish_data["w"], dish_data["h"]
                if auto_crop:
                    x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], padding, w, h)
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                preview = frame.copy()
                if dish_data["r"] > 0:
                    cv2.circle(preview, (dish_data["cx"], dish_data["cy"]), dish_data["r"], (82, 255, 168), 2)
                if not (x1 == 0 and y1 == 0 and x2 == w and y2 == h):
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 220, 82), 3)
                return x1, y1, x2, y2, preview

            vid_input.change(_on_video_upload, [vid_input],
                             [vid_dish_data, vid_frame_state, vid_cx1, vid_cy1, vid_cx2, vid_cy2,
                              vid_crop_preview, vid_trim_s, vid_trim_e])
            vid_auto_crop.change(_update_crop_preview, [vid_dish_data, vid_frame_state, vid_auto_crop, vid_pad],
                                 [vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_crop_preview])
            vid_pad.change(_update_crop_preview, [vid_dish_data, vid_frame_state, vid_auto_crop, vid_pad],
                           [vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_crop_preview])
            vid_btn.click(track_video,
                          inputs=[vid_input, vid_conf, vid_iou, vid_quality, vid_trail,
                                  vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_trim_s, vid_trim_e],
                          outputs=[vid_output, vid_stats, vid_csv, vid_analytics])

        # ================================================================
        # TAB 3: BATCH PROCESSING
        # ================================================================
        with gr.Tab("Batch Processing"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="section-label">Input</div>')
                    batch_input = gr.File(file_count="multiple", file_types=["image"],
                                         label="Upload Images (select multiple)")
                    gr.HTML('<div class="section-label">Settings</div>')
                    batch_quality = gr.Radio(["Fast", "Balanced", "Quality"], value=DEFAULT_QUALITY, label="Quality")
                    with gr.Row():
                        batch_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence")
                        batch_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05, label="NMS IoU")
                    batch_btn = gr.Button("Process All Images", variant="primary", size="lg")
                    batch_stats = gr.Markdown("*Upload images and click Process.*")
                    batch_zip = gr.File(label="Download ZIP")

                with gr.Column(scale=3):
                    gr.HTML('<div class="section-label">Results</div>')
                    batch_gallery = gr.Gallery(label="Detection Results", columns=4, height=600)

            batch_btn.click(process_batch,
                            inputs=[batch_input, batch_conf, batch_iou, batch_quality],
                            outputs=[batch_gallery, batch_stats, batch_zip])

        # ================================================================
        # TAB 4: FRAME EXPLORER
        # ================================================================
        with gr.Tab("Frame Explorer"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="section-label">Input</div>')
                    explorer_vid = gr.Video(label="Upload Video", height=220)
                    gr.HTML('<div class="section-label">Navigation</div>')
                    explorer_frame = gr.Slider(0, 1000, value=0, step=1, label="Frame", info="Scrub through video")
                    explorer_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence")

                    with gr.Accordion("Crop Settings", open=True):
                        exp_crop_msg = gr.Markdown("*Upload a video for auto-detection.*")
                        with gr.Row():
                            exp_auto_crop = gr.Checkbox(label="Auto-crop dish", value=True)
                            exp_pad = gr.Slider(-50, 150, value=50, step=5, label="Padding")
                        exp_cx1 = gr.Number(value=0, visible=False, precision=0)
                        exp_cy1 = gr.Number(value=0, visible=False, precision=0)
                        exp_cx2 = gr.Number(value=0, visible=False, precision=0)
                        exp_cy2 = gr.Number(value=0, visible=False, precision=0)

                    explorer_btn = gr.Button("Analyze Frame", variant="primary", size="lg")

                with gr.Column(scale=3):
                    gr.HTML('<div class="section-label">Analysis</div>')
                    explorer_output = gr.Image(label="Frame Analysis", height=560)
                    explorer_stats = gr.Markdown("*Select a frame and click Analyze.*")

            exp_dish_data = gr.State(value=None)
            exp_frame_state = gr.State(value=None)

            def _on_explorer_upload(video_path):
                if video_path is None:
                    return gr.Slider(maximum=1000), None, None, "*Upload a video.*", 0, 0, 0, 0
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return gr.Slider(maximum=1000), None, None, "Error.", 0, 0, 0, 0
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return gr.Slider(maximum=max(1, total-1)), None, None, "Error reading frame.", 0, 0, 0, 0
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dish = auto_detect_dish(frame)
                dish_data = {"cx": 0, "cy": 0, "r": 0, "w": w, "h": h}
                msg = "*No dish detected — set crop manually.*"
                if dish:
                    cx, cy, r = dish
                    dish_data = {"cx": cx, "cy": cy, "r": r, "w": w, "h": h}
                    msg = f"✓ Petri dish detected at ({cx}, {cy}) r={r}px"
                x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], 50, w, h)
                return gr.Slider(maximum=max(1, total-1)), dish_data, frame_rgb, msg, x1, y1, x2, y2

            def _update_exp_crop(dish_data, frame, auto_crop, padding):
                if dish_data is None:
                    return 0, 0, 0, 0
                w, h = dish_data["w"], dish_data["h"]
                if auto_crop:
                    x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], padding, w, h)
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                return x1, y1, x2, y2

            explorer_vid.change(_on_explorer_upload, [explorer_vid],
                                [explorer_frame, exp_dish_data, exp_frame_state, exp_crop_msg,
                                 exp_cx1, exp_cy1, exp_cx2, exp_cy2])
            exp_auto_crop.change(_update_exp_crop, [exp_dish_data, exp_frame_state, exp_auto_crop, exp_pad],
                                 [exp_cx1, exp_cy1, exp_cx2, exp_cy2])
            exp_pad.change(_update_exp_crop, [exp_dish_data, exp_frame_state, exp_auto_crop, exp_pad],
                           [exp_cx1, exp_cy1, exp_cx2, exp_cy2])
            explorer_btn.click(analyze_frame_by_frame,
                               inputs=[explorer_vid, explorer_conf, explorer_frame,
                                       exp_cx1, exp_cy1, exp_cx2, exp_cy2],
                               outputs=[explorer_output, explorer_stats])

        # ================================================================
        # TAB 5: VIDEO TOOLS
        # ================================================================
        with gr.Tab("Video Tools"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="section-label">Input</div>')
                    tools_input = gr.Video(label="Upload Video", height=220)
                    tools_info = gr.Markdown("*Upload a video to begin.*")

                    gr.HTML('<div class="section-label">Crop</div>')
                    with gr.Row():
                        tools_auto_crop = gr.Checkbox(label="Auto-detect dish", value=True)
                        tools_pad = gr.Slider(-50, 150, value=50, step=5, label="Padding")
                    crop_x1 = gr.Number(value=0, visible=False, precision=0)
                    crop_y1 = gr.Number(value=0, visible=False, precision=0)
                    crop_x2 = gr.Number(value=1920, visible=False, precision=0)
                    crop_y2 = gr.Number(value=1080, visible=False, precision=0)

                    gr.HTML('<div class="section-label">Trim (seconds)</div>')
                    with gr.Row():
                        trim_start = gr.Number(value=0, label="Start", precision=1)
                        trim_end = gr.Number(value=0, label="End (0=full)", precision=1)

                    tools_process_btn = gr.Button("Crop & Trim Video", variant="primary", size="lg")

                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.Tab("Preview"):
                            tools_preview = gr.Image(label="Crop Preview", height=480)
                        with gr.Tab("Processed Video"):
                            tools_output = gr.Video(label="Result", height=480)
                    with gr.Row():
                        with gr.Column(scale=2):
                            tools_stats = gr.Markdown("*Process a video to see stats.*")
                        with gr.Column(scale=1, min_width=160):
                            tools_download = gr.File(label="Download")

            tools_dish_data = gr.State(value=None)
            tools_frame_state = gr.State(value=None)

            def _on_tools_upload(video_path):
                if video_path is None:
                    return None, None, "*Upload a video.*", 0, 0, 1920, 1080, 0, 0, None
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None, None, "Error.", 0, 0, 1920, 1080, 0, 0, None
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur = total / fps
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None, None, "Error.", 0, 0, 1920, 1080, 0, 0, None
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dish = auto_detect_dish(frame)
                dish_data = {"cx": 0, "cy": 0, "r": 0, "w": w, "h": h}
                msg = f"**{w}×{h}, {dur:.1f}s** — No dish detected."
                if dish:
                    cx, cy, r = dish
                    dish_data = {"cx": cx, "cy": cy, "r": r, "w": w, "h": h}
                    msg = f"**{w}×{h}, {dur:.1f}s** — Dish detected ✓"
                x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], 50, w, h)
                preview = frame_rgb.copy()
                if dish:
                    cv2.circle(preview, (dish[0], dish[1]), dish[2], (82, 255, 168), 2)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 220, 82), 3)
                return dish_data, frame_rgb, msg, x1, y1, x2, y2, 0, round(dur, 1), preview

            def _update_tools_preview(dish_data, frame, auto_crop, padding):
                if dish_data is None or frame is None:
                    return 0, 0, 1920, 1080, None
                w, h = dish_data["w"], dish_data["h"]
                if auto_crop:
                    x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], padding, w, h)
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                preview = frame.copy()
                if dish_data["r"] > 0:
                    cv2.circle(preview, (dish_data["cx"], dish_data["cy"]), dish_data["r"], (82, 255, 168), 2)
                if not (x1 == 0 and y1 == 0 and x2 == w and y2 == h):
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 220, 82), 3)
                return x1, y1, x2, y2, preview

            tools_input.change(_on_tools_upload, [tools_input],
                               [tools_dish_data, tools_frame_state, tools_info,
                                crop_x1, crop_y1, crop_x2, crop_y2, trim_start, trim_end, tools_preview])
            tools_auto_crop.change(_update_tools_preview,
                                   [tools_dish_data, tools_frame_state, tools_auto_crop, tools_pad],
                                   [crop_x1, crop_y1, crop_x2, crop_y2, tools_preview])
            tools_pad.change(_update_tools_preview,
                             [tools_dish_data, tools_frame_state, tools_auto_crop, tools_pad],
                             [crop_x1, crop_y1, crop_x2, crop_y2, tools_preview])
            tools_process_btn.click(crop_and_trim_video,
                                    inputs=[tools_input, crop_x1, crop_y1, crop_x2, crop_y2, trim_start, trim_end],
                                    outputs=[tools_output, tools_stats, tools_download])

        # ================================================================
        # TAB 6: HELP
        # ================================================================
        with gr.Tab("Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
## System
| Component | Value |
|-----------|-------|
| Device | {HW['device']} ({HW.get('device_name', 'N/A')}) |
| OS | {HW['os']} {HW['arch']} |
| Speed | {HW['patch_time_ms']:.0f}ms per 416×416 patch |
| Default quality | {DEFAULT_QUALITY} |
| Model | YOLOv8m — 25.8M parameters |

## Quick Start

**Image Analysis** — Upload an image → set quality → click Analyze. Download CSV for coordinates.

**Video Tracking** — Upload video → auto-crop detects the dish → set quality → Start Tracking. Download CSV for per-track stats.

**Batch Processing** — Upload multiple images → process all at once → download ZIP with annotated images + CSV.

**Frame Explorer** — Step through a video frame-by-frame. Useful for debugging detection thresholds.

**Video Tools** — Pre-process videos (crop to dish region + trim) before tracking for faster inference.
""")
                with gr.Column():
                    gr.Markdown(f"""
## Quality Presets

| Preset | Overlap | Frame Skip | Best For |
|--------|---------|------------|----------|
| Fast | 25% | every 6th | Quick previews, slow hardware |
| Balanced | 40% | every 3rd | Daily use (recommended) |
| Quality | 50% | every frame | Publication-quality results |

## Tips for Accuracy
- **Crop to dish** first — fewer background pixels = higher accuracy + 2–10× faster
- **Confidence 0.3–0.4** for dim/small worms; **0.5–0.6** for clear high-contrast worms
- **Quality mode** catches ~10–15% more worms vs Fast (more patch overlap)
- **Batch + Quality** is the best combination for large datasets
- Video tracking uses centroid matching — works well for slow-moving worms

## Optimizations in This Build
- **Batch patch inference** on GPU: submits {8 if HW['device'] in ('CUDA', 'MPS') else 1} patches per inference call (vs 1)
- **CLAHE preprocessing** in dish detection for better contrast
- **Adaptive ETA** tracks real processing speed during video runs
- **Corner-mark drawing style** for cleaner visual output
""")

    gr.HTML(f"""
    <div class="footer-note">
        Worm Analyzer v2.0 &nbsp;·&nbsp; {HW['device']} ({HW.get('device_name','')}) &nbsp;·&nbsp;
        YOLOv8m sliding-window detection &nbsp;·&nbsp; {HW['patch_time_ms']:.0f}ms/patch
    </div>
    """)


# ============================================================================
# LAUNCH
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WORM ANALYZER v2.0")
    print("  http://localhost:7860")
    print("=" * 60 + "\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )