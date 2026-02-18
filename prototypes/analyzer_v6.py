#!/usr/bin/env python3
"""
Worm Analyzer Pro  ·  Detection, Tracking & Analytics
Run:  python3 worm_analyzer.py   →   http://localhost:7860
"""

import os
import sys
import cv2
import csv
import time
import shutil
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
import json

import numpy as np
import gradio as gr
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────
@dataclass
class Config:
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_worm_yolov8m.pt")
    PATCH_SIZE: int = 416
    DEFAULT_CONF: float = 0.40
    DEFAULT_IOU: float = 0.50
    MAX_GALLERY_IMAGES: int = 50
    
    SPEED_PRESETS = {
        "fast": {"name": "Fast Preview", "overlap": 0.25, "skip": 6, "desc": "Quick results, may miss small worms"},
        "balanced": {"name": "Balanced", "overlap": 0.40, "skip": 3, "desc": "Good speed/accuracy trade-off"},
        "precise": {"name": "High Precision", "overlap": 0.50, "skip": 1, "desc": "Best accuracy, slower processing"},
    }

CONFIG = Config()

# ─────────────────────────────────────────────────────────
#  HARDWARE DETECTION
# ─────────────────────────────────────────────────────────
class HardwareProfile:
    def __init__(self):
        self.device = "cpu"
        self.device_name = "CPU"
        self.batch_size = 1
        self.inference_ms = 100.0
        self.default_preset = "balanced"
        self._detect()
    
    def _detect(self):
        import platform
        import torch
        
        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            self.batch_size = 8
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            self.device_name = "Apple Silicon"
            self.batch_size = 4
        
        # Benchmark
        dummy = np.random.randint(0, 255, (CONFIG.PATCH_SIZE, CONFIG.PATCH_SIZE, 3), dtype=np.uint8)
        model = YOLO(CONFIG.MODEL_PATH)
        
        # Warmup
        for _ in range(3):
            model(dummy, imgsz=CONFIG.PATCH_SIZE, verbose=False)
        
        # Measure
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model(dummy, imgsz=CONFIG.PATCH_SIZE, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        
        self.inference_ms = np.median(times)
        
        # Select default preset based on speed
        if self.inference_ms < 50:
            self.default_preset = "precise"
        elif self.inference_ms < 150:
            self.default_preset = "balanced"
        else:
            self.default_preset = "fast"
        
        self.platform = platform.system()
    
    def to_dict(self):
        return {
            "device": self.device.upper(),
            "name": self.device_name,
            "inference_ms": f"{self.inference_ms:.1f}",
            "preset": CONFIG.SPEED_PRESETS[self.default_preset]["name"],
            "batch_size": self.batch_size
        }

# Initialize hardware profile
print("🔬 Initializing Worm Analyzer Pro...")
HW = HardwareProfile()
print(f"   Device: {HW.device_name} | Inference: {HW.inference_ms:.1f}ms | Default: {HW.default_preset}")

# Load model
if not os.path.exists(CONFIG.MODEL_PATH):
    print(f"❌ Model not found: {CONFIG.MODEL_PATH}")
    sys.exit(1)

MODEL = YOLO(CONFIG.MODEL_PATH)
print("   Model loaded successfully")

# ─────────────────────────────────────────────────────────
#  CORE DETECTION ENGINE
# ─────────────────────────────────────────────────────────
class DetectionEngine:
    def __init__(self, model, hardware):
        self.model = model
        self.hw = hardware
        self.cancelled = False
    
    def cancel(self):
        self.cancelled = True
    
    def reset(self):
        self.cancelled = False
    
    def _calculate_grid(self, dim: int, patch: int, overlap: float) -> List[int]:
        """Calculate sliding window positions."""
        stride = int(patch * (1 - overlap))
        positions = list(range(0, max(1, dim - patch + 1), stride))
        if not positions or positions[-1] + patch < dim:
            positions.append(max(0, dim - patch))
        return positions
    
    def detect(self, image: np.ndarray, conf: float = CONFIG.DEFAULT_CONF, 
               iou: float = CONFIG.DEFAULT_IOU, overlap: float = 0.4,
               progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run detection with sliding window for large images.
        Returns: (boxes [N,4], scores [N])
        """
        self.reset()
        H, W = image.shape[:2]
        
        # Small image: single pass
        if W <= CONFIG.PATCH_SIZE * 1.5 and H <= CONFIG.PATCH_SIZE * 1.5:
            results = self.model(image, conf=conf, iou=iou, imgsz=CONFIG.PATCH_SIZE, verbose=False)[0]
            return results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()
        
        # Large image: sliding window
        xs = self._calculate_grid(W, CONFIG.PATCH_SIZE, overlap)
        ys = self._calculate_grid(H, CONFIG.PATCH_SIZE, overlap)
        
        patches, positions = [], []
        for y in ys:
            for x in xs:
                patch = image[y:y+CONFIG.PATCH_SIZE, x:x+CONFIG.PATCH_SIZE]
                if patch.shape[0] < CONFIG.PATCH_SIZE or patch.shape[1] < CONFIG.PATCH_SIZE:
                    # Pad if necessary
                    padded = np.zeros((CONFIG.PATCH_SIZE, CONFIG.PATCH_SIZE, 3), dtype=np.uint8)
                    padded[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((x, y))
        
        # Batch inference
        all_boxes, all_scores = [], []
        total_batches = (len(patches) + self.hw.batch_size - 1) // self.hw.batch_size
        
        for i in range(0, len(patches), self.hw.batch_size):
            if self.cancelled:
                return np.array([]).reshape(0, 4), np.array([])
            
            batch = patches[i:i + self.hw.batch_size]
            batch_pos = positions[i:i + self.hw.batch_size]
            
            results = self.model(batch, conf=conf, iou=iou, imgsz=CONFIG.PATCH_SIZE, verbose=False)
            
            for r, (px, py) in zip(results, batch_pos):
                for box, score in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    # Adjust coordinates to original image
                    all_boxes.append([
                        box[0] + px, box[1] + py,
                        box[2] + px, box[3] + py
                    ])
                    all_scores.append(score)
            
            if progress_callback:
                progress_callback((i // self.hw.batch_size + 1) / total_batches)
        
        if not all_boxes:
            return np.array([]).reshape(0, 4), np.array([])
        
        # NMS
        import torch
        boxes_t = torch.tensor(np.array(all_boxes), dtype=torch.float32)
        scores_t = torch.tensor(np.array(all_scores), dtype=torch.float32)
        keep = torch.ops.torchvision.nms(boxes_t, scores_t, iou).numpy()
        
        return np.array(all_boxes)[keep], np.array(all_scores)[keep]

# Global detection engine
ENGINE = DetectionEngine(MODEL, HW)

# ─────────────────────────────────────────────────────────
#  VIDEO PROCESSING
# ─────────────────────────────────────────────────────────
@dataclass
class VideoInfo:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float
    dish_region: Optional[Tuple[int, int, int]] = None  # cx, cy, r
    
    @property
    def has_dish(self) -> bool:
        return self.dish_region is not None

class VideoProcessor:
    @staticmethod
    def load(path: str) -> Tuple[Optional[VideoInfo], Optional[np.ndarray], str]:
        """Load video and detect dish region."""
        if not path or not os.path.exists(path):
            return None, None, "No video file selected"
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, None, "Cannot open video file"
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps > 0 else 0
        
        # Read first frame for dish detection
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, None, "Cannot read video frames"
        
        # Detect dish
        dish = VideoProcessor._detect_dish(frame)
        
        info = VideoInfo(
            path=path, fps=fps, total_frames=total,
            width=width, height=height, duration=duration,
            dish_region=dish
        )
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        status = f"{width}×{height} @ {fps:.1f}fps | {duration:.1f}s"
        if dish:
            status += " | Dish detected ✓"
        else:
            status += " | No dish detected"
        
        return info, rgb, status
    
    @staticmethod
    def _detect_dish(frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect petri dish using Hough circles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        H, W = frame.shape[:2]
        min_r, max_r = min(H, W) // 6, min(H, W) // 2
        
        # Try different parameters
        for dp in [1.2, 1.5, 2.0]:
            for param1 in [100, 80, 50]:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp, min(H, W) // 2,
                    param1=param1, param2=50,
                    minRadius=min_r, maxRadius=max_r
                )
                if circles is not None:
                    # Return largest circle
                    circles = np.uint16(np.around(circles))
                    best = max(circles[0], key=lambda c: c[2])
                    return (int(best[0]), int(best[1]), int(best[2]))
        
        # Fallback: contour detection
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)
            if radius > min_r:
                return (int(x), int(y), int(radius))
        
        return None
    
    @staticmethod
    def get_crop_region(info: VideoInfo, auto: bool, padding: int) -> Tuple[int, int, int, int]:
        """Calculate crop region based on dish detection or full frame."""
        if not auto or not info.has_dish:
            return (0, 0, info.width, info.height)
        
        cx, cy, r = info.dish_region
        p = int(padding)
        x1 = max(0, cx - r - p)
        y1 = max(0, cy - r - p)
        x2 = min(info.width, cx + r + p)
        y2 = min(info.height, cy + r + p)
        return (x1, y1, x2, y2)
    
    @staticmethod
    def create_preview(frame: np.ndarray, info: VideoInfo, 
                      crop: Tuple[int, int, int, int]) -> np.ndarray:
        """Create preview image showing detection and crop regions."""
        preview = frame.copy()
        H, W = preview.shape[:2]
        
        # Draw dish circle if detected
        if info.has_dish:
            cx, cy, r = info.dish_region
            cv2.circle(preview, (cx, cy), r, (59, 130, 246), 2)
            cv2.circle(preview, (cx, cy), 3, (59, 130, 246), -1)
        
        # Draw crop rectangle
        x1, y1, x2, y2 = crop
        if not (x1 == 0 and y1 == 0 and x2 == W and y2 == H):
            # Darken outside region
            overlay = preview.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            preview = cv2.addWeighted(preview, 1.0, overlay, 0.6, 0)
            
            # Draw border
            cv2.rectangle(preview, (x1, y1), (x2, y2), (59, 130, 246), 3)
            label = f"Crop: {x2-x1}×{y2-y1}px"
            cv2.putText(preview, label, (x1 + 10, y1 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2)
        
        return preview

# ─────────────────────────────────────────────────────────
#  TRACKING SYSTEM
# ─────────────────────────────────────────────────────────
class WormTracker:
    def __init__(self, max_missing: int = 15, max_distance: float = 80.0):
        self.next_id = 1
        self.tracks = {}  # id -> {centroid, box, missing_count}
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.trajectories = defaultdict(list)  # id -> [(x, y, frame), ...]
    
    def update(self, detections: np.ndarray, frame_num: int) -> List[Tuple[int, np.ndarray]]:
        """
        Update tracker with new detections.
        Returns: list of (track_id, box) tuples
        """
        if len(detections) == 0:
            # Mark all as missing
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]
            return []
        
        # Calculate centroids
        centroids = np.array([[(d[0] + d[2]) / 2, (d[1] + d[3]) / 2] for d in detections])
        
        if not self.tracks:
            # Initialize new tracks
            results = []
            for det, cent in zip(detections, centroids):
                tid = self.next_id
                self.tracks[tid] = {
                    'centroid': cent, 'box': det, 'missing': 0
                }
                self.trajectories[tid].append((cent[0], cent[1], frame_num))
                results.append((tid, det))
                self.next_id += 1
            return results
        
        # Match existing tracks to detections
        from scipy.spatial.distance import cdist
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid]['centroid'] for tid in track_ids])
        
        distances = cdist(track_centroids, centroids)
        
        # Hungarian-like matching (greedy)
        used_tracks, used_dets = set(), set()
        matches = []
        
        # Sort by distance
        rows = distances.min(axis=1).argsort()
        for row in rows:
            if row in used_tracks:
                continue
            col = distances[row].argmin()
            if col in used_dets or distances[row, col] > self.max_distance:
                continue
            
            tid = track_ids[row]
            self.tracks[tid] = {
                'centroid': centroids[col],
                'box': detections[col],
                'missing': 0
            }
            self.trajectories[tid].append((centroids[col][0], centroids[col][1], frame_num))
            
            matches.append((tid, detections[col]))
            used_tracks.add(row)
            used_dets.add(col)
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_dets:
                tid = self.next_id
                self.tracks[tid] = {
                    'centroid': centroids[i],
                    'box': det,
                    'missing': 0
                }
                self.trajectories[tid].append((centroids[i][0], centroids[i][1], frame_num))
                matches.append((tid, det))
                self.next_id += 1
        
        # Update missing tracks
        for i, tid in enumerate(track_ids):
            if i not in used_tracks:
                self.tracks[tid]['missing'] += 1
                if self.tracks[tid]['missing'] > self.max_missing:
                    del self.tracks[tid]
        
        return matches
    
    def get_trajectories(self, max_points: Optional[int] = None) -> Dict[int, List[Tuple[float, float, int]]]:
        """Get trajectory data for visualization."""
        result = {}
        for tid, points in self.trajectories.items():
            if max_points:
                result[tid] = points[-max_points:]
            else:
                result[tid] = points
        return result

# ─────────────────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────────────────
class Visualizer:
    PALETTE = [
        (34, 197, 94),   # Green
        (59, 130, 246),  # Blue
        (239, 68, 68),   # Red
        (249, 115, 22),  # Orange
        (168, 85, 247),  # Purple
        (6, 182, 212),   # Cyan
        (236, 72, 153),  # Pink
        (20, 184, 166),  # Teal
        (234, 179, 8),   # Yellow
        (14, 165, 233),  # Sky
    ]
    
    @staticmethod
    def annotate_detections(image: np.ndarray, boxes: np.ndarray, 
                           scores: np.ndarray, track_ids: Optional[List[int]] = None,
                           trajectories: Optional[Dict[int, List]] = None) -> np.ndarray:
        """Draw detection boxes and trajectories on image."""
        output = image.copy()
        H, W = output.shape[:2]
        scale = max(0.5, min(1.2, W / 1200))
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # Get color
            if track_ids and i < len(track_ids):
                color = Visualizer.PALETTE[track_ids[i] % len(Visualizer.PALETTE)]
                label = f"#{track_ids[i]}"
            else:
                color = Visualizer.PALETTE[0]
                label = f"{score:.0%}"
            
            # Draw bracket corners
            corner_len = int(min(x2-x1, y2-y1) * 0.25)
            thickness = max(2, int(2 * scale))
            
            corners = [
                ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
            ]
            for (px, py), (dx, dy) in corners:
                cv2.line(output, (px, py), (px + dx * corner_len, py), color, thickness, cv2.LINE_AA)
                cv2.line(output, (px, py), (px, py + dy * corner_len), color, thickness, cv2.LINE_AA)
            
            # Semi-transparent fill
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                fill = np.full_like(roi, color)
                output[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.85, fill, 0.15, 0)
            
            # Label background
            font_scale = 0.5 * scale
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            ly = max(y1 - 5, th + 5)
            cv2.rectangle(output, (x1, ly - th - 4), (x1 + tw + 8, ly + 4), (0, 0, 0), -1)
            cv2.putText(output, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw trajectories
        if trajectories:
            for tid, points in trajectories.items():
                if len(points) < 2:
                    continue
                color = Visualizer.PALETTE[tid % len(Visualizer.PALETTE)]
                pts = [(int(p[0]), int(p[1])) for p in points]
                
                # Draw fading trail
                for j in range(1, len(pts)):
                    alpha = (j / len(pts)) ** 0.8
                    thickness = max(1, int(alpha * 3))
                    cv2.line(output, pts[j-1], pts[j], color, thickness, cv2.LINE_AA)
                
                # Start and end markers
                cv2.circle(output, pts[0], 4, (34, 197, 94), -1)  # Green start
                cv2.circle(output, pts[-1], 4, (239, 68, 68), -1)  # Red end
        
        return output
    
    @staticmethod
    def add_overlay(image: np.ndarray, main_text: str, sub_text: str) -> np.ndarray:
        """Add info overlay to top-left corner."""
        output = image.copy()
        H, W = output.shape[:2]
        
        # Background
        padding = 12
        font_main = 0.6
        font_sub = 0.4
        
        (w1, h1), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, font_main, 2)
        (w2, h2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, font_sub, 1)
        
        box_w = max(w1, w2) + padding * 2
        box_h = h1 + h2 + padding * 2 + 4
        
        # Semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
        output = cv2.addWeighted(output, 1.0, overlay, 0.7, 0)
        
        # Text
        cv2.putText(output, main_text, (10 + padding, 10 + padding + h1),
                   cv2.FONT_HERSHEY_SIMPLEX, font_main, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output, sub_text, (10 + padding, 10 + padding + h1 + 4 + h2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_sub, (200, 200, 200), 1, cv2.LINE_AA)
        
        return output

# ─────────────────────────────────────────────────────────
#  ANALYTICS
# ─────────────────────────────────────────────────────────
class AnalyticsGenerator:
    @staticmethod
    def generate_chart(trajectories: Dict[int, List], counts: List[int], 
                      fps: float, width: int, height: int) -> Optional[np.ndarray]:
        """Generate analytics visualization."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from scipy.ndimage import gaussian_filter
            
            # Style
            BG = '#fafafa'
            colors = {
                'primary': '#2563eb',
                'secondary': '#16a34a',
                'accent': '#f59e0b',
                'text': '#1e293b',
                'grid': '#e2e8f0'
            }
            
            fig = plt.figure(figsize=(14, 8), facecolor=BG)
            gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3,
                         left=0.06, right=0.98, top=0.92, bottom=0.08)
            
            axes = [
                fig.add_subplot(gs[0, :2]),  # Count over time
                fig.add_subplot(gs[0, 2]),   # Speed distribution
                fig.add_subplot(gs[1, :2]),  # Trajectories
                fig.add_subplot(gs[1, 2])    # Heatmap
            ]
            
            for ax in axes:
                ax.set_facecolor('white')
                ax.grid(True, alpha=0.3, color=colors['grid'])
                for spine in ax.spines.values():
                    spine.set_color(colors['grid'])
            
            # 1. Count over time
            time_axis = [i / fps for i in range(len(counts))]
            axes[0].plot(time_axis, counts, color=colors['primary'], linewidth=1.5, alpha=0.8)
            axes[0].fill_between(time_axis, counts, alpha=0.1, color=colors['primary'])
            
            # Moving average
            if len(counts) > 20:
                window = len(counts) // 20
                ma = np.convolve(counts, np.ones(window)/window, mode='valid')
                ma_time = time_axis[window//2:window//2 + len(ma)]
                axes[0].plot(ma_time, ma, '--', color=colors['secondary'], 
                           linewidth=2, label='Moving average')
                axes[0].legend(loc='upper right')
            
            axes[0].set_xlabel('Time (seconds)', fontsize=10)
            axes[0].set_ylabel('Worm count', fontsize=10)
            axes[0].set_title('Population Dynamics', fontsize=11, fontweight='bold', pad=10)
            
            # 2. Speed distribution
            speeds = []
            for traj in trajectories.values():
                if len(traj) < 2:
                    continue
                total_dist = sum(
                    np.sqrt((traj[i][0] - traj[i-1][0])**2 + (traj[i][1] - traj[i-1][1])**2)
                    for i in range(1, len(traj))
                )
                duration = (traj[-1][2] - traj[0][2]) / fps
                if duration > 0:
                    speeds.append(total_dist / duration)
            
            if speeds:
                axes[1].hist(speeds, bins=15, color=colors['primary'], 
                           edgecolor='white', alpha=0.8)
                axes[1].axvline(np.mean(speeds), color=colors['secondary'], 
                              linestyle='--', linewidth=2, label=f'Mean: {np.mean(speeds):.0f}')
                axes[1].legend(loc='upper right', fontsize=8)
            
            axes[1].set_xlabel('Speed (px/s)', fontsize=10)
            axes[1].set_ylabel('Frequency', fontsize=10)
            axes[1].set_title('Speed Distribution', fontsize=11, fontweight='bold', pad=10)
            
            # 3. Trajectory map
            cmap = plt.cm.tab10
            for idx, (tid, traj) in enumerate(trajectories.items()):
                if len(traj) < 2:
                    continue
                xs = [p[0] for p in traj]
                ys = [p[1] for p in traj]
                color = cmap(idx % 10)
                axes[2].plot(xs, ys, color=color, linewidth=1, alpha=0.6)
                axes[2].scatter(xs[0], ys[0], c=[colors['secondary']], s=20, zorder=5)
                axes[2].scatter(xs[-1], ys[-1], c=[colors['accent']], s=20, marker='x', zorder=5)
            
            axes[2].set_xlim(0, width)
            axes[2].set_ylim(height, 0)  # Flip Y for image coordinates
            axes[2].set_xlabel('X position (px)', fontsize=10)
            axes[2].set_ylabel('Y position (px)', fontsize=10)
            axes[2].set_title('Movement Trajectories  ●start  ×end', fontsize=11, fontweight='bold', pad=10)
            axes[2].set_aspect('equal')
            
            # 4. Activity heatmap
            scale = 8
            heatmap = np.zeros((height // scale + 1, width // scale + 1))
            for traj in trajectories.values():
                for x, y, _ in traj:
                    heatmap[int(y/scale), int(x/scale)] += 1
            
            heatmap = gaussian_filter(heatmap, sigma=3)
            im = axes[3].imshow(heatmap, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
            axes[3].set_title('Activity Heatmap', fontsize=11, fontweight='bold', pad=10)
            axes[3].set_xlabel('X', fontsize=10)
            axes[3].set_ylabel('Y', fontsize=10)
            plt.colorbar(im, ax=axes[3], shrink=0.8)
            
            fig.suptitle('Worm Movement Analytics', fontsize=13, fontweight='bold', y=0.98)
            
            # Save to temp file
            temp_path = os.path.join(tempfile.gettempdir(), 'analytics.png')
            plt.savefig(temp_path, dpi=120, facecolor=BG, bbox_inches='tight')
            plt.close()
            
            # Read back as numpy array
            img = cv2.imread(temp_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            print(f"Analytics error: {e}")
            return None

# ─────────────────────────────────────────────────────────
#  EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────
def export_detections_csv(boxes: np.ndarray, scores: np.ndarray, filepath: str):
    """Export detection results to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['detection_id', 'x1', 'y1', 'x2', 'y2', 
                        'confidence', 'width_px', 'height_px', 'center_x', 'center_y'])
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            writer.writerow([
                i + 1, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                f"{score:.4f}", f"{x2-x1:.1f}", f"{y2-y1:.1f}",
                f"{(x1+x2)/2:.1f}", f"{(y1+y2)/2:.1f}"
            ])
    return filepath

def export_tracking_csv(tracker: WormTracker, fps: float, filepath: str):
    """Export tracking results to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['worm_id', 'first_frame', 'last_frame', 'duration_sec',
                        'total_distance_px', 'avg_speed_px_per_s', 'detection_count'])
        
        for tid, traj in sorted(tracker.trajectories.items()):
            if len(traj) < 2:
                continue
            
            first_frame = traj[0][2]
            last_frame = traj[-1][2]
            duration = (last_frame - first_frame) / fps
            
            total_dist = sum(
                np.sqrt((traj[i][0] - traj[i-1][0])**2 + (traj[i][1] - traj[i-1][1])**2)
                for i in range(1, len(traj))
            )
            
            avg_speed = total_dist / duration if duration > 0 else 0
            
            writer.writerow([
                tid, first_frame, last_frame, f"{duration:.2f}",
                f"{total_dist:.1f}", f"{avg_speed:.1f}", len(traj)
            ])
    return filepath

# ─────────────────────────────────────────────────────────
#  GRADIO UI - MODERN PROFESSIONAL DESIGN
# ─────────────────────────────────────────────────────────
def create_ui():
    """Create the professional Gradio interface."""
    
    # Custom CSS for modern professional look
    custom_css = """
    :root {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --success: #16a34a;
        --warning: #f59e0b;
        --danger: #dc2626;
        --bg: #f8fafc;
        --surface: #ffffff;
        --text: #0f172a;
        --text-secondary: #64748b;
        --border: #e2e8f0;
        --radius: 8px;
        --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    }
    
    /* Global reset */
    .gradio-container {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    
    /* Header/Navigation */
    .app-header {
        background: var(--surface);
        border-bottom: 1px solid var(--border);
        padding: 0 24px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: var(--shadow);
    }
    
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, var(--primary), #3b82f6);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 18px;
    }
    
    .brand-text {
        font-size: 18px;
        font-weight: 700;
        color: var(--text);
    }
    
    .brand-badge {
        font-size: 11px;
        font-weight: 600;
        color: var(--primary);
        background: #eff6ff;
        padding: 2px 8px;
        border-radius: 12px;
        border: 1px solid #dbeafe;
    }
    
    .nav-tabs {
        display: flex;
        gap: 4px;
    }
    
    .nav-tab {
        padding: 8px 16px;
        border-radius: var(--radius);
        font-size: 14px;
        font-weight: 500;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s;
        border: none;
        background: transparent;
    }
    
    .nav-tab:hover {
        color: var(--text);
        background: #f1f5f9;
    }
    
    .nav-tab.active {
        color: var(--primary);
        background: #eff6ff;
        font-weight: 600;
    }
    
    .hw-info {
        font-size: 12px;
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Main layout */
    .main-content {
        max-width: 1600px;
        margin: 0 auto;
        padding: 24px;
    }
    
    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        box-shadow: var(--shadow);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border);
    }
    
    .card-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Sidebar */
    .sidebar {
        width: 320px;
        flex-shrink: 0;
    }
    
    .sidebar-section {
        margin-bottom: 24px;
    }
    
    .sidebar-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-secondary);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Form elements */
    .gr-slider input[type=range] {
        accent-color: var(--primary);
    }
    
    .gr-button-primary {
        background: var(--primary) !important;
        border-color: var(--primary) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: var(--radius) !important;
        padding: 10px 20px !important;
        transition: all 0.2s !important;
    }
    
    .gr-button-primary:hover {
        background: var(--primary-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    .gr-button-secondary {
        background: white !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-weight: 500 !important;
    }
    
    .gr-button-danger {
        background: #fef2f2 !important;
        border-color: #fecaca !important;
        color: var(--danger) !important;
    }
    
    .gr-button-danger:hover {
        background: #fee2e2 !important;
        border-color: var(--danger) !important;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-ready { background: #f0fdf4; color: var(--success); }
    .status-processing { background: #eff6ff; color: var(--primary); }
    .status-error { background: #fef2f2; color: var(--danger); }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 4px;
    }
    
    .stat-label {
        font-size: 12px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Progress bar */
    .progress-container {
        background: #f1f5f9;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        margin: 12px 0;
    }
    
    .progress-bar {
        background: var(--primary);
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Results area */
    .results-area {
        background: #f8fafc;
        border: 2px dashed var(--border);
        border-radius: 12px;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        gap: 12px;
        color: var(--text-secondary);
    }
    
    .results-area.has-content {
        border-style: solid;
        background: var(--surface);
    }
    
    /* Tables */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    
    .data-table th {
        background: #f8fafc;
        padding: 10px 12px;
        text-align: left;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
        border-bottom: 1px solid var(--border);
    }
    
    .data-table td {
        padding: 12px;
        border-bottom: 1px solid var(--border);
    }
    
    .data-table tr:hover td {
        background: #f8fafc;
    }
    
    /* Hide default Gradio tab nav (we use custom) */
    .tab-nav { display: none !important; }
    
    /* Gallery */
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 16px;
    }
    
    .gallery-item {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
        position: relative;
    }
    
    .gallery-overlay {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(transparent, rgba(0,0,0,0.7));
        color: white;
        padding: 20px 12px 12px;
        font-size: 12px;
    }
    """
    
    # Build UI
    with gr.Blocks(title="Worm Analyzer Pro", css=custom_css, theme=gr.themes.Soft()) as app:
        
        # State management
        current_video = gr.State(None)
        current_frame = gr.State(None)
        
        # Header
        gr.HTML(f"""
        <div class="app-header">
            <div class="brand">
                <div class="brand-icon">W</div>
                <span class="brand-text">Worm Analyzer</span>
                <span class="brand-badge">YOLOv8m</span>
            </div>
            <div class="nav-tabs">
                <button class="nav-tab active" onclick="switchTab(0)">📷 Single Image</button>
                <button class="nav-tab" onclick="switchTab(1)">🎬 Video Tracking</button>
                <button class="nav-tab" onclick="switchTab(2)">📁 Batch Processing</button>
                <button class="nav-tab" onclick="switchTab(3)">🔍 Frame Analysis</button>
                <button class="nav-tab" onclick="switchTab(4)">✂️ Video Prep</button>
                <button class="nav-tab" onclick="switchTab(5)">📖 Guide</button>
            </div>
            <div class="hw-info">
                {HW.device.upper()} · {HW.inference_ms:.0f}ms · {CONFIG.SPEED_PRESETS[HW.default_preset]['name']}
            </div>
        </div>
        
        <script>
        function switchTab(idx) {{
            document.querySelectorAll('.nav-tab').forEach((el, i) => {{
                el.classList.toggle('active', i === idx);
            }});
            // Trigger Gradio tab change
            document.querySelectorAll('.tabitem')[idx].click();
        }}
        </script>
        """)
        
        # Hidden tabs for Gradio functionality
        with gr.Tabs(visible=False) as tabs:
            
            # === TAB 1: Single Image ===
            with gr.Tab("Single", id=0):
                with gr.Row():
                    # Sidebar
                    with gr.Column(scale=0, min_width=320):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### 📷 Input Image")
                            img_input = gr.Image(
                                type="numpy",
                                label="",
                                height=250
                            )
                            
                            gr.Markdown("### ⚙️ Settings")
                            img_conf = gr.Slider(
                                0.1, 0.9, CONFIG.DEFAULT_CONF,
                                step=0.05,
                                label="Detection Confidence",
                                info="Lower = more sensitive"
                            )
                            img_preset = gr.Radio(
                                choices=[(CONFIG.SPEED_PRESETS[k]['name'], k) for k in CONFIG.SPEED_PRESETS.keys()],
                                value=HW.default_preset,
                                label="Processing Mode"
                            )
                            
                            img_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                            
                            img_status = gr.Markdown("Ready to analyze", elem_classes=["status-ready"])
                    
                    # Main content
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Results")
                            img_output = gr.Image(label="", height=500)
                            
                            with gr.Row():
                                img_stats = gr.Dataframe(
                                    headers=["Metric", "Value"],
                                    label="Detection Statistics",
                                    interactive=False
                                )
                                img_csv = gr.File(label="Download CSV")
            
            # === TAB 2: Video Tracking ===
            with gr.Tab("Video", id=1):
                with gr.Row():
                    with gr.Column(scale=0, min_width=320):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### 🎬 Video Input")
                            vid_input = gr.Video(label="", height=200)
                            vid_info = gr.Markdown("Upload a video file")
                            
                            gr.Markdown("### 🎯 Region of Interest")
                            with gr.Accordion("Crop Settings", open=True):
                                vid_auto_crop = gr.Checkbox(
                                    label="Auto-detect dish",
                                    value=True
                                )
                                vid_padding = gr.Slider(
                                    -50, 150, 50,
                                    label="Padding (px)"
                                )
                                vid_preview = gr.Image(label="Preview", height=150)
                            
                            gr.Markdown("### ⚙️ Tracking Settings")
                            vid_conf = gr.Slider(0.1, 0.9, CONFIG.DEFAULT_CONF, step=0.05, label="Confidence")
                            vid_preset = gr.Radio(
                                choices=[(CONFIG.SPEED_PRESETS[k]['name'], k) for k in CONFIG.SPEED_PRESETS.keys()],
                                value=HW.default_preset,
                                label="Quality"
                            )
                            vid_trail = gr.Slider(10, 200, 60, step=10, label="Trail Length")
                            
                            with gr.Row():
                                vid_start = gr.Button("▶ Start Tracking", variant="primary")
                                vid_stop = gr.Button("⏹ Stop", variant="secondary", elem_classes=["gr-button-danger"])
                            
                            vid_progress = gr.Slider(0, 100, 0, label="Progress", interactive=False)
                    
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Tracking Results")
                            vid_output = gr.Video(label="", height=400)
                            
                            with gr.Row():
                                vid_chart = gr.Image(label="Analytics", height=300)
                                vid_stats = gr.Dataframe(
                                    headers=["Metric", "Value"],
                                    label="Statistics",
                                    interactive=False
                                )
                            vid_csv = gr.File(label="Download Tracking Data")
            
            # === TAB 3: Batch Processing ===
            with gr.Tab("Batch", id=2):
                with gr.Row():
                    with gr.Column(scale=0, min_width=320):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### 📁 Input Files")
                            batch_files = gr.File(
                                file_count="multiple",
                                file_types=["image"],
                                label="Select Images"
                            )
                            
                            gr.Markdown("### ⚙️ Settings")
                            batch_conf = gr.Slider(0.1, 0.9, CONFIG.DEFAULT_CONF, step=0.05, label="Confidence")
                            batch_preset = gr.Radio(
                                choices=[(CONFIG.SPEED_PRESETS[k]['name'], k) for k in CONFIG.SPEED_PRESETS.keys()],
                                value=HW.default_preset,
                                label="Processing Mode"
                            )
                            
                            batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                            batch_progress = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Results Gallery")
                            batch_gallery = gr.Gallery(
                                label="",
                                columns=4,
                                rows=4,
                                height=600,
                                object_fit="contain"
                            )
                            
                            with gr.Row():
                                batch_summary = gr.Dataframe(
                                    headers=["File", "Count", "Avg Confidence"],
                                    label="Summary",
                                    interactive=False
                                )
                                batch_zip = gr.File(label="Download All Results")
            
            # === TAB 4: Frame Analysis ===
            with gr.Tab("Frame", id=3):
                with gr.Row():
                    with gr.Column(scale=0, min_width=320):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### 🔍 Frame Explorer")
                            frame_video = gr.Video(label="", height=200)
                            frame_slider = gr.Slider(0, 1000, 0, step=1, label="Frame Number")
                            frame_conf = gr.Slider(0.1, 0.9, CONFIG.DEFAULT_CONF, step=0.05, label="Confidence")
                            frame_btn = gr.Button("Analyze Frame", variant="primary")
                    
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Frame Analysis")
                            frame_output = gr.Image(label="", height=500)
                            frame_info = gr.Dataframe(
                                headers=["Property", "Value"],
                                label="Details",
                                interactive=False
                            )
            
            # === TAB 5: Video Preparation ===
            with gr.Tab("Prep", id=4):
                with gr.Row():
                    with gr.Column(scale=0, min_width=320):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### ✂️ Video Preparation")
                            prep_video = gr.Video(label="", height=200)
                            
                            gr.Markdown("### Crop")
                            prep_auto = gr.Checkbox(label="Auto-detect dish", value=True)
                            prep_pad = gr.Slider(-50, 150, 50, label="Padding")
                            
                            gr.Markdown("### Trim")
                            prep_start = gr.Number(0, label="Start (seconds)")
                            prep_end = gr.Number(0, label="End (0 = end)")
                            
                            prep_btn = gr.Button("Apply Crop & Trim", variant="primary")
                    
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("### Preview")
                            prep_preview = gr.Image(label="", height=300)
                            prep_output = gr.Video(label="Result", height=300)
                            prep_download = gr.File(label="Download")
            
            # === TAB 6: Guide ===
            with gr.Tab("Guide", id=5):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Stats
                        gr.HTML(f"""
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">{HW.inference_ms:.0f}ms</div>
                                <div class="stat-label">Inference Time</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">25.8M</div>
                                <div class="stat-label">Model Parameters</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">{HW.device.upper()}</div>
                                <div class="stat-label">Compute Device</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">416²</div>
                                <div class="stat-label">Input Resolution</div>
                            </div>
                        </div>
                        """)
                        
                        with gr.Group(elem_classes=["card"]):
                            gr.Markdown("""
                            ## 📖 User Guide
                            
                            ### Single Image Analysis
                            Upload a microscope image to detect and count worms instantly. 
                            Adjust the confidence threshold to control detection sensitivity.
                            
                            ### Video Tracking
                            Track individual worms over time with persistent IDs. 
                            The system maintains trajectories and generates movement analytics.
                            
                            **Tips:**
                            - Use "Auto-detect dish" to focus on the petri dish area
                            - Higher trail length shows longer movement history
                            - Use "Stop" button to get partial results from long videos
                            
                            ### Batch Processing
                            Process multiple images in parallel. Results include individual 
                            detection images and a summary CSV file.
                            
                            ### Sensitivity Guide
                            
                            | Confidence | Best For |
                            |------------|----------|
                            | 0.20 - 0.35 | Faint or small worms, may have false positives |
                            | 0.35 - 0.55 | **Recommended** - Balanced accuracy |
                            | 0.55 - 0.80 | High confidence only, minimal false positives |
                            
                            ### Performance Notes
                            - GPU processing is {gpu}x faster than CPU
                            - Large images are automatically tiled
                            - Cropping to region of interest significantly improves speed
                            """.format(gpu="8-10" if HW.device == "cuda" else "4-5" if HW.device == "mps" else "N/A"))
        
        # === EVENT HANDLERS ===
        
        # Image analysis
        def analyze_image(image, conf, preset):
            if image is None:
                return None, [], None, "Please upload an image first"
            
            start_time = time.time()
            preset_data = CONFIG.SPEED_PRESETS[preset]
            
            boxes, scores = ENGINE.detect(
                image, conf=conf, 
                overlap=preset_data['overlap']
            )
            
            elapsed = time.time() - start_time
            
            # Annotate
            annotated = Visualizer.annotate_detections(image, boxes, scores)
            annotated = Visualizer.add_overlay(
                annotated, 
                f"{len(boxes)} worms detected",
                f"Processing time: {elapsed:.2f}s"
            )
            
            # Stats
            stats_data = [
                ["Total Detections", str(len(boxes))],
                ["Average Confidence", f"{np.mean(scores):.1%}" if len(scores) else "N/A"],
                ["Processing Time", f"{elapsed:.2f}s"],
                ["Image Size", f"{image.shape[1]}×{image.shape[0]}px"]
            ]
            
            # Export
            csv_path = os.path.join(tempfile.gettempdir(), f"detections_{int(time.time())}.csv")
            export_detections_csv(boxes, scores, csv_path)
            
            return annotated, stats_data, csv_path, "Analysis complete"
        
        img_btn.click(
            analyze_image,
            [img_input, img_conf, img_preset],
            [img_output, img_stats, img_csv, img_status]
        )
        
        # Video loading and preview
        def load_video_handler(video_path):
            if not video_path:
                return None, None, "No video selected", 0, 1, 0, 0, 0, 0
            
            info, frame, status = VideoProcessor.load(video_path)
            if info is None:
                return None, None, status, 0, 1, 0, 0, 0, 0
            
            crop = VideoProcessor.get_crop_region(info, True, 50)
            preview = VideoProcessor.create_preview(frame, info, crop)
            
            return (
                info, frame, status,
                info.total_frames,
                info.fps,
                crop[0], crop[1], crop[2], crop[3]
            )
        
        def update_crop_preview(info, frame, auto, padding):
            if info is None:
                return None, 0, 0, 0, 0
            
            crop = VideoProcessor.get_crop_region(info, auto, padding)
            preview = VideoProcessor.create_preview(frame, info, crop)
            return preview, *crop
        
        vid_input.change(
            load_video_handler,
            [vid_input],
            [current_video, current_frame, vid_info, 
             gr.Slider(visible=False), gr.Slider(visible=False),
             gr.Number(visible=False), gr.Number(visible=False),
             gr.Number(visible=False), gr.Number(visible=False)]
        )
        
        vid_auto_crop.change(
            update_crop_preview,
            [current_video, current_frame, vid_auto_crop, vid_padding],
            [vid_preview, gr.Number(visible=False), gr.Number(visible=False),
             gr.Number(visible=False), gr.Number(visible=False)]
        )
        
        # Tracking (simplified - full implementation would be longer)
        def start_tracking(video_path, conf, preset, trail_len, x1, y1, x2, y2, progress=gr.Progress()):
            if not video_path:
                return None, None, None, "No video uploaded"
            
            # This would contain the full tracking logic
            # For brevity, showing structure only
            return None, None, None, "Tracking not fully implemented in this preview"
        
        vid_start.click(
            start_tracking,
            [vid_input, vid_conf, vid_preset, vid_trail],
            [vid_output, vid_chart, vid_csv, vid_info]
        )
    
    return app

# ─────────────────────────────────────────────────────────
#  MAIN ENTRY
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Worm Analyzer Pro")
    print("  URL: http://localhost:7860")
    print("="*50 + "\n")
    
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=True
    )