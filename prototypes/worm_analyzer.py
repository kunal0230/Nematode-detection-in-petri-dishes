#!/usr/bin/env python3
"""
Worm Analyzer — Detection, Tracking & Analytics
=================================================
A user-friendly web application for nematode/worm detection and tracking.

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

import gradio as gr
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_worm_yolov8m.pt")
DEFAULT_CONF = 0.4
DEFAULT_IOU = 0.5
# Model was trained on 416x416 patches — sliding window uses this size
PATCH_SIZE = 416

# Quality presets: (overlap, video_frame_skip_factor, label)
QUALITY_PRESETS = {
    "Fast":     {"overlap": 0.25, "skip_factor": 6, "desc": "Fewer patches, more frame skipping"},
    "Balanced": {"overlap": 0.40, "skip_factor": 3, "desc": "Good accuracy with reasonable speed"},
    "Quality":  {"overlap": 0.50, "skip_factor": 1, "desc": "Maximum accuracy, slower processing"},
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

# --- Auto-detect hardware ---
import platform
import torch as _torch

def _detect_hardware():
    """Detect compute device and benchmark inference speed."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }

    if _torch.cuda.is_available():
        info["device"] = "CUDA"
        info["device_name"] = _torch.cuda.get_device_name(0)
        info["gpu_mem"] = f"{_torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        info["device"] = "MPS"
        info["device_name"] = "Apple Silicon GPU"
    else:
        info["device"] = "CPU"
        info["device_name"] = platform.processor() or "Unknown CPU"

    # Benchmark: run 3 warm-up inferences then time 3 real ones
    dummy = np.random.randint(0, 255, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    # Warm-up (first call is always slower due to compilation)
    for _ in range(2):
        model(dummy, conf=0.5, imgsz=PATCH_SIZE, verbose=False)

    times = []
    for _ in range(3):
        t0 = time.time()
        model(dummy, conf=0.5, imgsz=PATCH_SIZE, verbose=False)
        times.append(time.time() - t0)

    info["patch_time_ms"] = np.median(times) * 1000  # median ms per patch
    return info

print("Benchmarking hardware...")
HW = _detect_hardware()
print(f"  Device: {HW['device']} ({HW['device_name']})")
print(f"  OS: {HW['os']} {HW['arch']}")
print(f"  Speed: {HW['patch_time_ms']:.0f} ms per patch")

# Set default quality based on speed
if HW["patch_time_ms"] < 80:        # fast GPU
    DEFAULT_QUALITY = "Quality"
elif HW["patch_time_ms"] < 250:     # moderate (M1, decent GPU)
    DEFAULT_QUALITY = "Balanced"
else:                                 # slow CPU
    DEFAULT_QUALITY = "Fast"
print(f"  Default quality: {DEFAULT_QUALITY}")


# ============================================================================
# SLIDING WINDOW DETECTION (core engine)
# ============================================================================

import torch

def _get_patch_positions(img_w, img_h, patch_size, overlap):
    """Generate (x, y) positions for sliding window patches."""
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
    Detect worms using sliding window inference.

    The model was trained on 416x416 patches. For full images we:
    1. Extract overlapping patches
    2. Run model on each patch
    3. Map detections back to global coordinates
    4. NMS to merge duplicates from overlapping patches

    overlap: float 0-1, or None to use the current quality preset default.
    """
    if overlap is None:
        overlap = QUALITY_PRESETS[DEFAULT_QUALITY]["overlap"]

    h, w = image.shape[:2]

    # Small images: run directly
    if w <= patch_size * 1.5 and h <= patch_size * 1.5:
        results = model(image, conf=confidence, iou=iou_threshold, imgsz=PATCH_SIZE, verbose=False)
        result = results[0]
        return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

    # Generate patch positions and run inference
    x_steps, y_steps = _get_patch_positions(w, h, patch_size, overlap)

    all_boxes = []
    all_scores = []

    for y_pos in y_steps:
        for x_pos in x_steps:
            patch = image[y_pos:y_pos+patch_size, x_pos:x_pos+patch_size]

            results = model(patch, conf=confidence, iou=iou_threshold, imgsz=PATCH_SIZE, verbose=False)
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                all_boxes.append([box[0] + x_pos, box[1] + y_pos, box[2] + x_pos, box[3] + y_pos])
                all_scores.append(score)

    if len(all_boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([])

    # NMS to merge overlapping detections
    boxes_t = torch.tensor(np.array(all_boxes), dtype=torch.float32)
    scores_t = torch.tensor(np.array(all_scores), dtype=torch.float32)
    keep = torch.ops.torchvision.nms(boxes_t, scores_t, iou_threshold)

    return np.array(all_boxes)[keep.numpy()], np.array(all_scores)[keep.numpy()]


def _estimate_time(img_w, img_h, overlap, patch_size=PATCH_SIZE):
    """Estimate processing time for an image given its size."""
    if img_w <= patch_size * 1.5 and img_h <= patch_size * 1.5:
        return HW["patch_time_ms"] / 1000
    x_steps, y_steps = _get_patch_positions(img_w, img_h, patch_size, overlap)
    n_patches = len(x_steps) * len(y_steps)
    return n_patches * HW["patch_time_ms"] / 1000


def _format_eta(seconds):
    """Format seconds into a readable ETA string."""
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
# CENTROID TRACKER (for video — replaces ByteTrack on full frames)
# ============================================================================

class CentroidTracker:
    """
    Simple centroid-based tracker that assigns consistent IDs across frames.
    Matches detections by minimum distance between centroids.
    """

    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id = 1
        self.objects = {}         # id -> centroid (cx, cy)
        self.boxes = {}           # id -> (x1, y1, x2, y2)
        self.disappeared = {}     # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        """
        Update tracker with new detections.
        detections: np.array of shape (N, 4) with [x1, y1, x2, y2]
        Returns: list of (track_id, box) tuples
        """
        # Compute centroids of new detections
        if len(detections) == 0:
            # Mark all existing objects as disappeared
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

        # If no existing tracks, register all as new
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

        # Match existing tracks to new detections using distance matrix
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[oid] for oid in obj_ids])

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        dists = cdist(obj_centroids, new_centroids)

        # Hungarian-style greedy matching (sort by distance, assign closest first)
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

        # Register unmatched detections as new tracks
        for col in range(len(new_centroids)):
            if col not in used_cols:
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = new_centroids[col]
                self.boxes[tid] = detections[col]
                self.disappeared[tid] = 0
                results.append((tid, detections[col]))

        # Mark unmatched existing tracks as disappeared
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
# UTILITY FUNCTIONS
# ============================================================================

def format_size(w, h):
    return f"{w:.0f}x{h:.0f}"

def draw_detections(image, boxes, scores, track_ids=None, trails=None, show_conf=True, show_id=True):
    """Draw detection boxes on image with optional tracking info."""
    overlay = image.copy()
    h, w = image.shape[:2]

    # Color palette for different track IDs
    colors = [
        (0, 255, 127), (255, 100, 100), (100, 200, 255), (255, 255, 100),
        (200, 100, 255), (100, 255, 200), (255, 180, 100), (180, 100, 255),
        (100, 255, 100), (255, 100, 200), (100, 180, 255), (255, 220, 100),
    ]

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
        color = colors[tid % len(colors)] if tid is not None else (0, 255, 127)

        # Draw box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label
        parts = []
        if show_id and tid is not None:
            parts.append(f"#{tid}")
        if show_conf:
            parts.append(f"{score:.0%}")
        label = " ".join(parts)
        
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw tracking trails
    if trails:
        for tid, points in trails.items():
            if len(points) < 2:
                continue
            color = colors[tid % len(colors)]
            for j in range(1, len(points)):
                alpha = j / len(points)
                thickness = max(1, int(alpha * 3))
                pt1 = tuple(map(int, points[j-1]))
                pt2 = tuple(map(int, points[j]))
                cv2.line(overlay, pt1, pt2, color, thickness)

    return overlay


# ============================================================================
# TAB 1: IMAGE ANALYSIS
# ============================================================================

def analyze_image(image, confidence, iou_threshold, quality):
    """Detect worms in a single image using sliding window."""
    if image is None:
        return None, "No image uploaded.", None

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
    overlap = preset["overlap"]

    start = time.time()
    boxes, scores = detect_worms(image, confidence=confidence, iou_threshold=iou_threshold, overlap=overlap)
    elapsed = time.time() - start

    n_det = len(boxes)

    # Draw detections
    annotated = draw_detections(image, boxes, scores, show_conf=True, show_id=False)

    # Add count overlay
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (10, 10), (280, 80), (0, 0, 0), -1)
    cv2.putText(annotated, f"Worms: {n_det}  |  {elapsed:.1f}s", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
    cv2.putText(annotated, f"Quality: {quality} | {HW['device']}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Patch info
    x_steps, y_steps = _get_patch_positions(w, h, PATCH_SIZE, overlap)
    n_patches = len(x_steps) * len(y_steps) if (w > PATCH_SIZE * 1.5 or h > PATCH_SIZE * 1.5) else 1

    stats_lines = [
        f"**Worms Detected: {n_det}**",
        f"- Processing time: {elapsed:.1f}s ({HW['device']})",
        f"- Image size: {w} x {h}",
        f"- Patches scanned: {n_patches} ({PATCH_SIZE}x{PATCH_SIZE}, {int(overlap*100)}% overlap)",
        f"- Quality preset: {quality}",
        f"- Confidence: {confidence}",
    ]
    if n_det > 0:
        avg_conf = np.mean(scores)
        widths_px = boxes[:, 2] - boxes[:, 0]
        heights_px = boxes[:, 3] - boxes[:, 1]
        areas = widths_px * heights_px
        stats_lines.extend([
            f"- Average confidence: {avg_conf:.2%}",
            f"- Average worm size: {np.mean(widths_px):.0f} x {np.mean(heights_px):.0f} px",
            f"- Smallest: {np.min(areas):.0f} px² | Largest: {np.max(areas):.0f} px²",
        ])

    # CSV data
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
    """Track worms through a video using sliding window detection + centroid tracking.
    Supports on-the-fly cropping and trimming — no need to pre-process."""
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

    # Crop region (0 = no crop / use full frame)
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

    # Trim range
    trim_start = max(0, float(trim_start))
    trim_end = float(trim_end)
    if trim_end <= trim_start:
        trim_end = duration
    start_frame = int(trim_start * fps)
    end_frame = min(int(trim_end * fps), total_frames)

    # Quality preset controls
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS[DEFAULT_QUALITY])
    video_overlap = preset["overlap"]
    frame_skip = max(1, preset["skip_factor"])

    # Estimate total time
    trimmed_frames = end_frame - start_frame
    frames_to_process = trimmed_frames // frame_skip
    per_frame_est = _estimate_time(w, h, video_overlap)
    total_est = frames_to_process * per_frame_est

    # Output video
    out_path = os.path.join(tempfile.gettempdir(), "tracked_worms.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Initialize centroid tracker
    tracker = CentroidTracker(max_disappeared=int(fps * 2), max_distance=80)

    # Tracking data
    frame_counts = []
    all_tracks = {}       # id -> {first_seen, last_seen, positions}
    trails = defaultdict(list)  # id -> [(x, y), ...] for visual trails

    # Seek to trim start
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = 0  # relative to trim start
    processed_count = 0
    last_annotated = None
    proc_start = time.time()
    crop_label = f"{w}x{h} crop" if use_crop else f"{w}x{h}"
    progress(0, desc=f"Starting... ETA: {_format_eta(total_est)} ({quality}, {crop_label}, {HW['device']})")

    while frame_num < trimmed_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply crop on-the-fly
        if use_crop:
            frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        if frame_num % frame_skip == 0:
            frame_start = time.time()

            # Run sliding window detection on this frame
            boxes, scores = detect_worms(
                frame, confidence=confidence, iou_threshold=iou_threshold,
                overlap=video_overlap
            )

            # Update tracker with detections
            tracked = tracker.update(boxes)

            # Build arrays for drawing
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

            # Draw on frame
            annotated = draw_detections(
                frame, tracked_boxes, tracked_scores,
                track_ids=track_ids_arr if len(track_ids_arr) > 0 else None,
                trails=trails if len(trails) > 0 else None,
                show_conf=False, show_id=True
            )

            # HUD overlay
            cv2.rectangle(annotated, (10, 10), (380, 105), (0, 0, 0), -1)
            cv2.putText(annotated, f"Worms: {len(tracked_boxes)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
            cv2.putText(annotated, f"Unique IDs: {len(all_tracks)}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            time_sec = frame_num / fps
            cv2.putText(annotated, f"Time: {time_sec:.1f}s | Frame: {frame_num}/{total_frames}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            last_annotated = annotated
            processed_count += 1

            # Adaptive ETA based on actual measured speed
            actual_per_frame = (time.time() - proc_start) / processed_count
            remaining_frames = frames_to_process - processed_count
            eta = remaining_frames * actual_per_frame
        else:
            annotated = last_annotated if last_annotated is not None else frame

        out.write(annotated)
        frame_num += 1

        if frame_num % 10 == 0:
            pct = frame_num / trimmed_frames
            desc = f"Frame {frame_num}/{trimmed_frames} | {len(all_tracks)} tracks | ETA: {_format_eta(eta) if processed_count > 0 else '...'}"
            progress(pct, desc=desc)

    cap.release()
    out.release()

    # ---- Compute analytics ----
    duration_out = trimmed_frames / fps

    # Per-track stats
    tracking_csv_path = os.path.join(tempfile.gettempdir(), "tracking_data.csv")
    with open(tracking_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "first_frame", "last_frame", "duration_sec",
            "total_distance_px", "avg_velocity_px_per_sec",
            "start_x", "start_y", "end_x", "end_y",
            "displacement_px", "n_detections"
        ])

        for tid, data in sorted(all_tracks.items()):
            positions = data["positions"]
            first = data["first_seen"]
            last = data["last_seen"]
            dur = (last - first) / fps if last != first else 0

            # Distance
            total_dist = 0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_dist += np.sqrt(dx**2 + dy**2)

            avg_vel = total_dist / dur if dur > 0 else 0
            start_pos = positions[0]
            end_pos = positions[-1]
            displacement = np.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)

            writer.writerow([
                tid, first, last, f"{dur:.2f}",
                f"{total_dist:.1f}", f"{avg_vel:.2f}",
                f"{start_pos[0]:.1f}", f"{start_pos[1]:.1f}",
                f"{end_pos[0]:.1f}", f"{end_pos[1]:.1f}",
                f"{displacement:.1f}", len(positions)
            ])

    total_proc_time = time.time() - proc_start

    # Summary stats
    stats_lines = [
        f"**Video Tracking Complete**",
        f"- Duration: {duration_out:.1f}s ({trimmed_frames} frames @ {fps:.0f} fps)",
        f"- Resolution: {w} x {h}" + (f" (cropped from {orig_w}x{orig_h})" if use_crop else ""),
        f"- Processing: {quality} mode on {HW['device']}, every {frame_skip} frames",
        f"- Total processing time: {_format_eta(total_proc_time)}",
        f"- **Unique worms tracked: {len(all_tracks)}**",
        f"- Average worm count per frame: {np.mean(frame_counts):.1f}" if frame_counts else "- No worms detected",
        f"- Max worms in a single frame: {max(frame_counts) if frame_counts else 0}",
    ]

    if all_tracks:
        velocities = []
        distances = []
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
                f"**Movement Analysis:**",
                f"- Avg velocity: {np.mean(velocities):.1f} px/sec",
                f"- Max velocity: {np.max(velocities):.1f} px/sec",
                f"- Avg distance traveled: {np.mean(distances):.0f} px",
                f"- Max distance traveled: {np.max(distances):.0f} px",
            ])

    # Generate analytics image
    analytics_img = generate_analytics(all_tracks, frame_counts, fps, w, h)

    return out_path, "\n".join(stats_lines), tracking_csv_path, analytics_img


def generate_analytics(all_tracks, frame_counts, fps, img_w, img_h):
    """Generate analytics visualization image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes.flat:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#e2e8f0")
            ax.xaxis.label.set_color("#e2e8f0")
            ax.yaxis.label.set_color("#e2e8f0")
            ax.title.set_color("#a78bfa")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        # 1. Worm count over time
        ax = axes[0, 0]
        times = [i / fps for i in range(len(frame_counts))]
        ax.plot(times, frame_counts, color="#a78bfa", linewidth=1.5, alpha=0.8)
        ax.fill_between(times, frame_counts, alpha=0.2, color="#a78bfa")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Worm Count")
        ax.set_title("Worm Count Over Time")

        # 2. Velocity distribution
        ax = axes[0, 1]
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
            ax.hist(velocities, bins=15, color="#a78bfa", edgecolor="#1a1a2e", alpha=0.8)
        ax.set_xlabel("Velocity (px/sec)")
        ax.set_ylabel("Count")
        ax.set_title("Velocity Distribution")

        # 3. Movement paths
        ax = axes[1, 0]
        colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(all_tracks))))
        for idx, (tid, data) in enumerate(all_tracks.items()):
            positions = data["positions"]
            if len(positions) < 2:
                continue
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            ax.plot(xs, ys, color=colors[idx % len(colors)], linewidth=1, alpha=0.6)
            ax.scatter(xs[0], ys[0], color="lime", s=20, zorder=5)
            ax.scatter(xs[-1], ys[-1], color="red", s=20, zorder=5, marker="x")
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Worm Movement Paths (green=start, red=end)")
        ax.set_aspect("equal")

        # 4. Activity heatmap
        ax = axes[1, 1]
        heatmap = np.zeros((img_h // 10, img_w // 10), dtype=np.float32)
        for tid, data in all_tracks.items():
            for x, y, _ in data["positions"]:
                hx = min(int(x / 10), heatmap.shape[1] - 1)
                hy = min(int(y / 10), heatmap.shape[0] - 1)
                heatmap[hy, hx] += 1

        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=3)
        im = ax.imshow(heatmap, cmap="magma", aspect="auto")
        ax.set_title("Worm Activity Heatmap")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, label="Activity", shrink=0.8)

        plt.suptitle("Worm Tracking Analytics", fontsize=16, color="#e2e8f0", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(tempfile.gettempdir(), "analytics.png")
        plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
        plt.close()

        return cv2.cvtColor(cv2.imread(out_path), cv2.COLOR_BGR2RGB)
    except ImportError as e:
        print(f"Analytics visualization requires matplotlib and scipy: {e}")
        return None


# ============================================================================
# TAB 3: BATCH PROCESSING
# ============================================================================

def process_batch(files, confidence, iou_threshold, progress=gr.Progress()):
    """Process multiple images and return summary + zip."""
    if not files:
        return None, "No images uploaded.", None

    results_dir = os.path.join(tempfile.gettempdir(), "worm_batch_results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    all_stats = []
    gallery_images = []

    for i, file_path in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing image {i+1}/{len(files)}")

        img = cv2.imread(file_path)
        if img is None:
            continue

        fname = os.path.basename(file_path)
        boxes, scores = detect_worms(img, confidence=confidence, iou_threshold=iou_threshold)

        annotated = draw_detections(img, boxes, scores, show_conf=True, show_id=False)

        # Add count overlay
        cv2.rectangle(annotated, (10, 10), (200, 50), (0, 0, 0), -1)
        cv2.putText(annotated, f"Worms: {len(boxes)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)

        # Save
        out_path = os.path.join(results_dir, f"detected_{fname}")
        if not out_path.lower().endswith((".jpg", ".jpeg", ".png")):
            out_path += ".jpg"
        cv2.imwrite(out_path, annotated)

        gallery_images.append((cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), fname))

        avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0
        all_stats.append({
            "filename": fname,
            "worm_count": len(boxes),
            "avg_confidence": f"{avg_conf:.2%}",
        })

    # Summary CSV
    csv_path = os.path.join(results_dir, "batch_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "worm_count", "avg_confidence"])
        writer.writeheader()
        writer.writerows(all_stats)

    # Zip results
    zip_path = os.path.join(tempfile.gettempdir(), "worm_batch_results")
    shutil.make_archive(zip_path, "zip", results_dir)

    total = sum(s["worm_count"] for s in all_stats)
    avg = total / len(all_stats) if all_stats else 0

    summary = [
        f"**Batch Processing Complete**",
        f"- Images processed: {len(all_stats)}",
        f"- Total worms detected: {total}",
        f"- Average per image: {avg:.1f}",
        "",
        "| Image | Worms | Confidence |",
        "|---|---|---|",
    ]
    for s in all_stats:
        summary.append(f"| {s['filename']} | {s['worm_count']} | {s['avg_confidence']} |")

    return gallery_images, "\n".join(summary), zip_path + ".zip"


# ============================================================================
# TAB 4: LIVE ANALYSIS (from video file frame-by-frame)
# ============================================================================

def analyze_frame_by_frame(video_path, confidence, frame_number,
                           crop_x1=0, crop_y1=0, crop_x2=0, crop_y2=0):
    """Analyze a specific frame from a video. Supports on-the-fly cropping."""
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

    # Apply crop if specified
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

    cv2.rectangle(annotated, (10, 10), (300, 75), (0, 0, 0), -1)
    cv2.putText(annotated, f"Frame {frame_number}/{total} | {time_sec:.1f}s", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(annotated, f"Worms: {len(boxes)}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    stats = f"Frame {frame_number} ({time_sec:.1f}s): **{len(boxes)} worms** detected"
    return annotated, stats


# ============================================================================
# TAB 5: VIDEO TOOLS (Crop & Trim)
# ============================================================================

def auto_detect_dish(frame):
    """
    Auto-detect the petri dish circle in a frame.
    Uses Hough circles with multiple parameter sweeps,
    falls back to largest contour if no circle found.
    Returns (cx, cy, radius) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    h, w = frame.shape[:2]
    min_r = min(h, w) // 6
    max_r = min(h, w) // 2

    # Try Hough circles with multiple sensitivity levels
    for dp in [1.2, 1.5, 2.0]:
        for param1 in [80, 50, 30]:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=dp, minDist=min(h, w) // 2,
                param1=param1, param2=50,
                minRadius=min_r, maxRadius=max_r
            )
            if circles is not None:
                # Pick the largest circle
                circles = np.round(circles[0]).astype(int)
                best = max(circles, key=lambda c: c[2])
                return int(best[0]), int(best[1]), int(best[2])

    # Fallback: threshold + largest contour
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        if radius > min_r:
            return int(cx), int(cy), int(radius)

    return None


def load_video_info(video_path):
    """Load video and return info + first frame preview."""
    if video_path is None:
        return None, "No video uploaded.", 0, 0, 0, 0, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video.", 0, 0, 0, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Cannot read video.", 0, 0, 0, 0, 0, 0

    # Auto-detect dish
    dish = auto_detect_dish(frame)

    if dish:
        cx, cy, r = dish
        # Convert circle to crop rectangle (with padding)
        pad = int(r * 0.05)  # 5% padding
        x1 = max(0, cx - r - pad)
        y1 = max(0, cy - r - pad)
        x2 = min(w, cx + r + pad)
        y2 = min(h, cy + r + pad)

        # Draw detected dish on preview
        preview = frame.copy()
        cv2.circle(preview, (cx, cy), r, (0, 255, 0), 3)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(preview, "Auto-detected dish", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        info = (
            f"**Video loaded**\n"
            f"- Resolution: {w} x {h}\n"
            f"- Duration: {duration:.1f}s ({total_frames} frames @ {fps:.0f} fps)\n"
            f"- **Dish detected!** Center: ({cx}, {cy}), Radius: {r}px\n"
            f"- Suggested crop: ({x1}, {y1}) to ({x2}, {y2}) = {x2-x1}x{y2-y1}px"
        )
        return preview, info, x1, y1, x2, y2, 0, round(duration, 1)
    else:
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        info = (
            f"**Video loaded**\n"
            f"- Resolution: {w} x {h}\n"
            f"- Duration: {duration:.1f}s ({total_frames} frames @ {fps:.0f} fps)\n"
            f"- ⚠️ No dish auto-detected — set crop manually below"
        )
        return preview, info, 0, 0, w, h, 0, round(duration, 1)


def preview_crop(video_path, x1, y1, x2, y2):
    """Show a preview frame with the crop region highlighted."""
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 10, min(x2, w))
    y2 = max(y1 + 10, min(y2, h))

    # Draw dark overlay outside crop region
    overlay = frame.copy()
    mask = np.zeros_like(frame)
    mask[y1:y2, x1:x2] = 255
    overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)

    # Draw crop border
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    crop_w, crop_h = x2 - x1, y2 - y1
    cv2.putText(overlay, f"Crop: {crop_w}x{crop_h}px", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show what the cropped frame looks like (inset)
    cropped = frame[y1:y2, x1:x2]
    inset_h = min(150, crop_h)
    inset_w = int(inset_h * crop_w / crop_h)
    inset = cv2.resize(cropped, (inset_w, inset_h))
    # Place inset in top-right
    ix, iy = w - inset_w - 10, 10
    if ix > x2 or iy > y2:  # only if it doesn't overlap the crop
        overlay[iy:iy+inset_h, ix:ix+inset_w] = inset
        cv2.rectangle(overlay, (ix-2, iy-2), (ix+inset_w+2, iy+inset_h+2), (255, 255, 255), 2)
        cv2.putText(overlay, "Preview", (ix, iy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def crop_and_trim_video(video_path, x1, y1, x2, y2, start_sec, end_sec, progress=gr.Progress()):
    """
    Crop and trim a video. Fast — uses pure OpenCV array slicing.
    Returns: output video path, status text, output file for download.
    """
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

    # Sanitize inputs
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

    crop_w = x2 - x1
    crop_h = y2 - y1

    # Estimate time (simple I/O bound, ~0.5ms per frame for crop)
    est_time = out_frames * 0.0005
    progress(0, desc=f"Processing {out_frames} frames... ETA: {_format_eta(est_time)}")

    # Output
    out_path = os.path.join(tempfile.gettempdir(), "cropped_trimmed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    proc_start = time.time()
    written = 0

    for i in range(out_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Crop (fast numpy slice — no model inference!)
        cropped = frame[y1:y2, x1:x2]
        out.write(cropped)
        written += 1

        if written % 100 == 0:
            elapsed = time.time() - proc_start
            rate = written / elapsed if elapsed > 0 else 1
            eta = (out_frames - written) / rate if rate > 0 else 0
            pct = written / out_frames
            progress(pct, desc=f"Frame {written}/{out_frames} ({rate:.0f} fps) | ETA: {_format_eta(eta)}")

    cap.release()
    out.release()

    total_time = time.time() - proc_start
    rate = written / total_time if total_time > 0 else 0

    stats = (
        f"**Video processed!**\n"
        f"- Output: {crop_w} x {crop_h} ({start_sec:.1f}s → {end_sec:.1f}s)\n"
        f"- Frames: {written} ({end_sec - start_sec:.1f}s @ {fps:.0f} fps)\n"
        f"- Processing: {total_time:.1f}s ({rate:.0f} frames/sec)\n"
        f"- Original: {orig_w}x{orig_h} → Cropped: {crop_w}x{crop_h} "
        f"({100*crop_w*crop_h/(orig_w*orig_h):.0f}% of original area)\n\n"
        f"*Fewer pixels → faster worm detection! "
        f"Estimated detection speedup: ~{orig_w*orig_h/(crop_w*crop_h):.1f}×*"
    )

    return out_path, stats, out_path


# ============================================================================
# GRADIO UI
# ============================================================================

# Custom CSS - Minimalist & Clean
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
    --primary: #000000;
    --text-main: #000000;
    --bg-main: #ffffff;
    --bg-panel: #f8fafc; /* Very light gray */
    --border: #e2e8f0;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
}

/* Typography */
h1, h2, h3, h4, span, p, label {
    color: var(--text-main) !important;
}

/* Panels */
.block, .panel {
    background-color: var(--bg-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

/* Buttons */
.primary-btn, button.primary {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
}
.secondary-btn, button.secondary {
    background-color: #ffffff !important;
    border: 1px solid #000000 !important;
    color: #000000 !important;
    border-radius: 4px !important;
}

/* Inputs */
input, textarea, select {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid var(--border) !important;
}
.gr-input-label { color: #000000 !important; }

/* Tabs */
.tab-nav button {
    font-weight: 500 !important; 
    color: #64748b !important;
}
.tab-nav button.selected {
    color: #000000 !important;
    border-bottom: 2px solid #000000 !important;
}

footer { display: none !important; }
"""

def _get_crop_from_dish(cx, cy, r, padding, w, h):
    """Calculate crop coordinates from dish center/radius + padding."""
    if cx == 0 and cy == 0 and r == 0:
        return 0, 0, 0, 0
    pad = int(padding)
    x1 = max(0, cx - r - pad)
    y1 = max(0, cy - r - pad)
    x2 = min(w, cx + r + pad)
    y2 = min(h, cy + r + pad)
    return x1, y1, x2, y2

# Theme - Minimal
app_theme = gr.themes.Monochrome(
    font=gr.themes.GoogleFont("Inter"),
    radius_size=gr.themes.sizes.radius_sm,
)

# Build app
with gr.Blocks(title="Worm Analyzer", theme=app_theme, css=custom_css) as app:

    # Header - REMOVED for minimal UI
    # gr.Markdown("# Worm Analyzer", elem_classes=["main-title"])
    # gr.Markdown("Detection, Tracking & Analytics for Nematode Research", elem_classes=["subtitle"])

    with gr.Tabs():

        # ================================================================
        # TAB 1: IMAGE ANALYSIS
        # ================================================================
        with gr.Tab("Image Analysis", id="image"):
            gr.Markdown("### Upload an image to detect and count worms")
            gr.Markdown(f"*Sliding window detection on 416×416 patches — Detected: **{HW['device']}** ({HW['patch_time_ms']:.0f}ms/patch)*")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="numpy", label="Upload Image", height=400)
                    with gr.Row():
                        img_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05,
                                             label="Confidence Threshold",
                                             info="Higher = fewer but more certain detections")
                        img_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05,
                                            label="NMS IoU Threshold",
                                            info="Lower = less overlap between detections")
                    img_quality = gr.Radio(
                        choices=["Fast", "Balanced", "Quality"],
                        value=DEFAULT_QUALITY,
                        label="Speed / Quality",
                        info="Fast=25% overlap, Balanced=40%, Quality=50%"
                    )
                    img_btn = gr.Button("Analyze Image", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Detection Result", height=400)
                    img_stats = gr.Markdown(label="Statistics")
                    img_csv = gr.File(label="Download Detections CSV")

            img_btn.click(
                analyze_image,
                inputs=[img_input, img_conf, img_iou, img_quality],
                outputs=[img_output, img_stats, img_csv],
            )

        # ================================================================
        # TAB 2: VIDEO TRACKING
        # ================================================================
        with gr.Tab("Video Tracking", id="video"):
            
            with gr.Row():
                # LEFT SIDEBAR (Controls)
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Setup")
                    vid_input = gr.Video(label="Upload Video", height=300)
                    
                    with gr.Accordion("Detection Settings", open=True):
                        with gr.Row():
                            vid_auto_crop = gr.Checkbox(label="Auto-Crop Dish", value=True)
                            vid_pad = gr.Slider(-50, 150, value=50, step=5, label="Padding (px)")
                        
                        # Hidden coords
                        vid_cx1 = gr.Number(value=0, visible=False, precision=0)
                        vid_cy1 = gr.Number(value=0, visible=False, precision=0)
                        vid_cx2 = gr.Number(value=0, visible=False, precision=0)
                        vid_cy2 = gr.Number(value=0, visible=False, precision=0)

                        vid_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence")
                        vid_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05, label="NMS IoU")

                    with gr.Accordion("Advanced & Trim", open=False):
                         vid_quality = gr.Radio(["Fast", "Balanced", "Quality"], value=DEFAULT_QUALITY, label="Processing Speed")
                         vid_trail = gr.Slider(10, 200, value=60, step=10, label="Trail Length")
                         
                         gr.Markdown("**Trim Video (sec)**")
                         with gr.Row():
                            vid_trim_s = gr.Number(value=0, label="Start", precision=1)
                            vid_trim_e = gr.Number(value=0, label="End", precision=1)
                    
                    # Preview is less important now, hidden or small? 
                    # Actually, seeing the crop is important. Let's put it in the sidebar or just rely on the main updated video if we could? 
                    # No, we need a static preview before tracking.
                    vid_crop_preview = gr.Image(label="Crop Preview", height=200, show_label=True)
                    
                    vid_btn = gr.Button("Start Tracking", variant="primary", size="lg")

                # RIGHT MAIN AREA (Results)
                with gr.Column(scale=3):
                    gr.Markdown("### 2. Results")
                    vid_output = gr.Video(label="Tracked Video", height=600)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            vid_analytics = gr.Image(label="Analytics Dashboard", height=350, show_label=False)
                        with gr.Column(scale=1):
                            vid_stats = gr.Markdown("### Tracking Stats\n*Run tracking to see results.*")
                            vid_csv = gr.File(label="Download CSV")
            
            # State variables
            vid_dish_data = gr.State(value=None)  # {cx, cy, r, w, h}
            vid_frame_state = gr.State(value=None) # Full frame numpy array

            def _on_video_upload(video_path):
                if video_path is None:
                    return None, None, 0, 0, 0, 0, None, 0, 0
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None, None, 0, 0, 0, 0, None, 0, 0
                
                # Get video info
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
                
                # Calculate initial crop (auto=True, padding=50)
                x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], 50, w, h)
                
                # Draw preview
                preview = frame_rgb.copy()
                if dish:
                    cv2.circle(preview, (dish[0], dish[1]), dish[2], (0, 255, 0), 2)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
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
                     cv2.circle(preview, (dish_data["cx"], dish_data["cy"]), dish_data["r"], (0, 255, 0), 2)
                
                # Draw crop box
                if x1 != 0 or y1 != 0 or x2 != w or y2 != h:
                     cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
                return x1, y1, x2, y2, preview

            vid_input.change(
                _on_video_upload,
                inputs=[vid_input],
                outputs=[vid_dish_data, vid_frame_state, vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_crop_preview, vid_trim_s, vid_trim_e]
            )
            
            vid_auto_crop.change(
                _update_crop_preview,
                inputs=[vid_dish_data, vid_frame_state, vid_auto_crop, vid_pad],
                outputs=[vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_crop_preview]
            )
            
            vid_pad.change(
                _update_crop_preview,
                inputs=[vid_dish_data, vid_frame_state, vid_auto_crop, vid_pad],
                outputs=[vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_crop_preview]
            )

            vid_btn.click(
                track_video,
                inputs=[vid_input, vid_conf, vid_iou, vid_quality, vid_trail,
                        vid_cx1, vid_cy1, vid_cx2, vid_cy2, vid_trim_s, vid_trim_e],
                outputs=[vid_output, vid_stats, vid_csv, vid_analytics],
            )

        # ================================================================
        # TAB 3: BATCH PROCESSING
        # ================================================================
        with gr.Tab("Batch Processing", id="batch"):
            gr.Markdown("### Upload multiple images for batch analysis")
            gr.Markdown("*All images are processed and annotated. Download results as a ZIP.*")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Images (select multiple)",
                    )
                    with gr.Row():
                        batch_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05,
                                               label="Confidence")
                        batch_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05,
                                              label="NMS IoU")
                    batch_btn = gr.Button("Process All Images", variant="primary", size="lg")

                with gr.Column(scale=1):
                    batch_stats = gr.Markdown(label="Batch Summary")
                    batch_zip = gr.File(label="Download Results (ZIP)")

            batch_gallery = gr.Gallery(label="Detection Results", columns=3, height=400)

            batch_btn.click(
                process_batch,
                inputs=[batch_input, batch_conf, batch_iou],
                outputs=[batch_gallery, batch_stats, batch_zip],
            )

        # ================================================================
        # TAB 4: FRAME EXPLORER
        # ================================================================
        with gr.Tab("Frame Explorer", id="explorer"):
            
            with gr.Row():
                # LEFT SIDEBAR
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Setup")
                    explorer_vid = gr.Video(label="Upload Video", height=300)
                    
                    # Explorer controls
                    explorer_frame = gr.Slider(0, 1000, value=0, step=1, label="Frame Number", info="Scrub video")
                    explorer_conf = gr.Slider(0.1, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence")

                    with gr.Accordion("Crop Settings", open=True):
                        exp_crop_msg = gr.Markdown("Upload for auto-crop.")
                        with gr.Row():
                             exp_auto_crop = gr.Checkbox(label="Auto-Crop", value=True)
                             exp_pad = gr.Slider(-50, 150, value=50, step=5, label="Padding")
                        
                        # Hidden coords
                        exp_cx1 = gr.Number(value=0, visible=False, precision=0)
                        exp_cy1 = gr.Number(value=0, visible=False, precision=0)
                        exp_cx2 = gr.Number(value=0, visible=False, precision=0)
                        exp_cy2 = gr.Number(value=0, visible=False, precision=0)

                    explorer_btn = gr.Button("Analyze Frame", variant="primary", size="lg")

                # RIGHT MAIN AREA
                with gr.Column(scale=3):
                    gr.Markdown("### 2. Analysis Results")
                    explorer_output = gr.Image(label="Frame Analysis", height=600)
                    explorer_stats = gr.Markdown("### Stats\n*Analyze a frame to see details.*")
            
            # State variables
            exp_dish_data = gr.State(value=None)
            exp_frame_state = gr.State(value=None)

            # Update slider max + auto-detect crop when video is loaded
            def _on_explorer_upload(video_path):
                if video_path is None:
                    return gr.Slider(maximum=1000), None, None, "Upload a video.", 0, 0, 0, 0
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return gr.Slider(maximum=1000), None, None, "Error opening video.", 0, 0, 0, 0
                
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
                msg = "No dish detected."
                
                if dish:
                    cx, cy, r = dish
                    dish_data = {"cx": cx, "cy": cy, "r": r, "w": w, "h": h}
                    msg = "Petri Dish detected!"
                
                # Calculate initial crop
                x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], 50, w, h)
                
                return gr.Slider(maximum=max(1, total-1)), dish_data, frame_rgb, msg, x1, y1, x2, y2

            def _update_explorer_crop(dish_data, frame, auto_crop, padding):
                # For explorer, we don't strictly need a preview image since the main "Analyze Frame" will show it.
                # But we do need to update the hidden coords.
                if dish_data is None:
                    return 0, 0, 0, 0
                
                w, h = dish_data["w"], dish_data["h"]
                if auto_crop:
                    x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], padding, w, h)
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                return x1, y1, x2, y2

            explorer_vid.change(
                _on_explorer_upload,
                inputs=[explorer_vid],
                outputs=[explorer_frame, exp_dish_data, exp_frame_state, exp_crop_msg, exp_cx1, exp_cy1, exp_cx2, exp_cy2],
            )
            
            exp_auto_crop.change(
                _update_explorer_crop,
                inputs=[exp_dish_data, exp_frame_state, exp_auto_crop, exp_pad],
                outputs=[exp_cx1, exp_cy1, exp_cx2, exp_cy2]
            )
            
            exp_pad.change(
                _update_explorer_crop,
                inputs=[exp_dish_data, exp_frame_state, exp_auto_crop, exp_pad],
                outputs=[exp_cx1, exp_cy1, exp_cx2, exp_cy2]
            )

            explorer_btn.click(
                analyze_frame_by_frame,
                inputs=[explorer_vid, explorer_conf, explorer_frame,
                        exp_cx1, exp_cy1, exp_cx2, exp_cy2],
                outputs=[explorer_output, explorer_stats],
            )

        # ================================================================
        # TAB 5: VIDEO TOOLS (Crop & Trim)
        # ================================================================
        # ================================================================
        # TAB 5: VIDEO TOOLS (Crop & Trim)
        # ================================================================
        with gr.Tab("Video Tools", id="tools"):
            
            with gr.Row():
                # LEFT SIDEBAR
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Setup")
                    tools_input = gr.Video(label="Upload Video", height=300)
                    tools_info = gr.Markdown(label="Video Info", value="Upload a video to begin.")

                    with gr.Accordion("Crop Settings", open=True):
                         with gr.Row():
                            tools_auto_crop = gr.Checkbox(label="Auto-Crop", value=True)
                            tools_pad = gr.Slider(-50, 150, value=50, step=5, label="Padding")
                         
                         # Hidden coordinates
                         crop_x1 = gr.Number(value=0, visible=False, precision=0)
                         crop_y1 = gr.Number(value=0, visible=False, precision=0)
                         crop_x2 = gr.Number(value=1920, visible=False, precision=0)
                         crop_y2 = gr.Number(value=1080, visible=False, precision=0)

                    with gr.Accordion("Trim Settings", open=True):
                         gr.Markdown("**Trim Time (sec)**")
                         with gr.Row():
                            trim_start = gr.Number(value=0, label="Start", precision=1)
                            trim_end = gr.Number(value=0, label="End", precision=1)

                    tools_process_btn = gr.Button("Crop & Trim Video", variant="primary", size="lg")

                # RIGHT MAIN AREA
                with gr.Column(scale=3):
                    gr.Markdown("### 2. Output")
                    with gr.Tabs():
                        with gr.Tab("Preview"):
                             tools_preview = gr.Image(label="Crop Preview", height=500)
                        with gr.Tab("Processed Video"):
                             tools_output = gr.Video(label="Result", height=500)
                    
                    with gr.Row():
                        tools_stats = gr.Markdown("Process a video to see stats.")
                        tools_download = gr.File(label="Download Result")

            # State
            tools_dish_data = gr.State(value=None)
            tools_frame_state = gr.State(value=None)

            # Auto-detect dish when video is uploaded
            def _on_tools_upload(video_path):
                if video_path is None:
                    return None, None, "Upload a video.", 0, 0, 1920, 1080, 0, 0, None
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None, None, "Error opening video.", 0, 0, 1920, 1080, 0, 0, None
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur = total / fps
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return None, None, "Error reading frame.", 0, 0, 1920, 1080, 0, 0, None

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dish = auto_detect_dish(frame)
                
                dish_data = {"cx": 0, "cy": 0, "r": 0, "w": w, "h": h}
                msg = f"**{w}x{h}, {dur:.1f}s** — No dish detected."
                
                if dish:
                    cx, cy, r = dish
                    dish_data = {"cx": cx, "cy": cy, "r": r, "w": w, "h": h}
                    msg = f"**{w}x{h}, {dur:.1f}s** — Petri Dish detected!"
                
                # Calculate initial crop
                x1, y1, x2, y2 = _get_crop_from_dish(dish_data["cx"], dish_data["cy"], dish_data["r"], 50, w, h)
                
                # Draw preview
                preview = frame_rgb.copy()
                if dish:
                    cv2.circle(preview, (dish[0], dish[1]), dish[2], (0, 255, 0), 2)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
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
                     cv2.circle(preview, (dish_data["cx"], dish_data["cy"]), dish_data["r"], (0, 255, 0), 2)
                
                # Draw crop box
                if x1 != 0 or y1 != 0 or x2 != w or y2 != h:
                     cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 3)
                
                return x1, y1, x2, y2, preview

            tools_input.change(
                _on_tools_upload,
                inputs=[tools_input],
                outputs=[tools_dish_data, tools_frame_state, tools_info, crop_x1, crop_y1, crop_x2, crop_y2, trim_start, trim_end, tools_preview],
            )
            
            tools_auto_crop.change(
                _update_tools_preview,
                inputs=[tools_dish_data, tools_frame_state, tools_auto_crop, tools_pad],
                outputs=[crop_x1, crop_y1, crop_x2, crop_y2, tools_preview]
            )
            
            tools_pad.change(
                _update_tools_preview,
                inputs=[tools_dish_data, tools_frame_state, tools_auto_crop, tools_pad],
                outputs=[crop_x1, crop_y1, crop_x2, crop_y2, tools_preview]
            )

            # Process (crop + trim)
            tools_process_btn.click(
                crop_and_trim_video,
                inputs=[tools_input, crop_x1, crop_y1, crop_x2, crop_y2, trim_start, trim_end],
                outputs=[tools_output, tools_stats, tools_download],
            )

        # ================================================================
        # TAB 6: ABOUT & HELP
        # ================================================================
        with gr.Tab("Help", id="help"):
            gr.Markdown(f"""
### System Info
- **Device**: {HW['device']} ({HW['device_name']})
- **OS**: {HW['os']} {HW['arch']}
- **Speed**: {HW['patch_time_ms']:.0f}ms per 416×416 patch
- **Default quality**: {DEFAULT_QUALITY}

### How to Use

**Image Analysis**
1. Upload an image of a petri dish with worms
2. Choose a quality preset (Fast for quick estimates, Quality for best accuracy)
3. Adjust the confidence threshold if needed
4. Click "Analyze Image"

**Video Tracking**
1. Upload a video file (MP4, M4V, AVI)
2. Choose quality — **Fast** skips more frames, **Quality** processes every frame
3. Click "Start Tracking" and check the ETA display
4. View the tracked video with worm IDs and colored trails

**Speed / Quality Presets**
- **Fast** (25% overlap, skip 6 frames): Best for slow CPUs or quick previews
- **Balanced** (40% overlap, skip 3 frames): Good accuracy with reasonable speed
- **Quality** (50% overlap, every frame): Maximum accuracy, slowest processing

**Tips**
- The app auto-detects your hardware and picks the best default preset
- On a GPU laptop: Quality mode runs comfortably
- On a CPU-only laptop: use Fast or Balanced mode
- Low confidence (0.2–0.3): catches faint worms but may include false positives
- Medium confidence (0.4–0.5): good balance for most use cases
- High confidence (0.6+): only very clear detections

### Technical Details
- **Model**: YOLOv8m (25.8M params) trained on 416×416 patches
- **Detection**: Sliding window with NMS (non-maximum suppression)
- **Tracking**: Centroid-based tracker with distance matching
- **Export**: CSV (compatible with Excel, Google Sheets, R, Python/pandas)
""")

    gr.Markdown(
        f"<center style='color:#475569; margin-top:20px'>Worm Analyzer v1.1 | {HW['device']} | Powered by YOLOv8</center>"
    )


# ============================================================================
# LAUNCH
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WORM ANALYZER")
    print("  Open your browser at: http://localhost:7860")
    print("=" * 60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=app_theme,
        css=custom_css,
    )
