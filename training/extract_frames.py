#!/usr/bin/env python3
"""
Extract Frames from Video
=========================
Extracts frames from a video file at a specified interval.
Designed for creating training data from worm observation videos.
"""

import cv2
import os
import sys

def extract_frames(video_path, output_dir='raw_data/images', interval_sec=20):
    """
    Extract frames from a video at a given interval.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    output_dir : str
        Directory to save extracted frames.
    interval_sec : int
        Interval in seconds between extracted frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    frame_interval = int(fps * interval_sec)

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print(f"  Extracting every {interval_sec}s ({frame_interval} frames)")
    print(f"  Expected frames: ~{int(duration_sec / interval_sec) + 1}")
    print()

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps
            filename = f"frame_{saved_count:03d}_t{int(timestamp_sec)}s.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"  Saved: {filename} (t={timestamp_sec:.1f}s)")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"\nDone. Extracted {saved_count} frames to {output_dir}/")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        extract_frames(video_path, interval_sec=interval)
    else:
        print("Usage: python3 extract_frames.py <video_path> [interval_seconds]")
