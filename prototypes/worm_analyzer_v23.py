#!/usr/bin/env python3
"""
Worm Analyzer — Detection, Tracking & Analytics
Launch:  python3 worm_analyzer.py  →  http://localhost:7860
"""

import os, sys, cv2, csv, time, shutil, tempfile
import numpy as np
from pathlib import Path
from collections import defaultdict

import gradio as gr
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_worm_yolov8m.pt")
DEFAULT_CONF = 0.40
DEFAULT_IOU  = 0.50
PATCH_SIZE   = 416

QUALITY = {
    "Fast":     {"overlap": 0.25, "skip": 6},
    "Balanced": {"overlap": 0.40, "skip": 3},
    "Quality":  {"overlap": 0.50, "skip": 1},
}

# ── Model + Hardware ──────────────────────────────────────────────────────────
print("Loading model…")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

model = YOLO(MODEL_PATH)

import platform, torch as _torch

def _detect_hw():
    hw = {"os": platform.system(), "arch": platform.machine()}
    if _torch.cuda.is_available():
        hw["device"] = "CUDA"
        hw["name"]   = _torch.cuda.get_device_name(0)
    elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        hw["device"] = "MPS"
        hw["name"]   = "Apple Silicon GPU"
    else:
        hw["device"] = "CPU"
        hw["name"]   = platform.processor() or "CPU"

    dummy = np.random.randint(0, 255, (PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    for _ in range(2): model(dummy, conf=0.5, imgsz=PATCH_SIZE, verbose=False)
    t = []
    for _ in range(3):
        t0 = time.time()
        model(dummy, conf=0.5, imgsz=PATCH_SIZE, verbose=False)
        t.append(time.time() - t0)
    hw["ms"] = np.median(t) * 1000
    return hw

print("Benchmarking…")
HW = _detect_hw()
DEFAULT_Q = "Quality" if HW["ms"] < 80 else ("Balanced" if HW["ms"] < 250 else "Fast")
print(f"  {HW['device']} · {HW['ms']:.0f}ms/patch · default={DEFAULT_Q}")

import torch

# ── Detection engine ──────────────────────────────────────────────────────────
def _patch_positions(W, H, patch, overlap):
    stride = int(patch * (1 - overlap))
    def steps(dim):
        s = list(range(0, max(1, dim - patch + 1), stride))
        if not s or s[-1] + patch < dim:
            s.append(max(0, dim - patch))
        return s
    return steps(W), steps(H)

def detect(image, conf=DEFAULT_CONF, iou=DEFAULT_IOU, overlap=None):
    if overlap is None:
        overlap = QUALITY[DEFAULT_Q]["overlap"]
    H, W = image.shape[:2]
    if W <= PATCH_SIZE * 1.5 and H <= PATCH_SIZE * 1.5:
        r = model(image, conf=conf, iou=iou, imgsz=PATCH_SIZE, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    xs, ys = _patch_positions(W, H, PATCH_SIZE, overlap)
    patches, positions = [], []
    for y in ys:
        for x in xs:
            patches.append(image[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            positions.append((x, y))

    batch = 8 if HW["device"] in ("CUDA", "MPS") else 1
    boxes_all, scores_all = [], []
    for i in range(0, len(patches), batch):
        results = model(patches[i:i+batch], conf=conf, iou=iou, imgsz=PATCH_SIZE, verbose=False)
        for r, (px, py) in zip(results, positions[i:i+batch]):
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            for box, sc in zip(b, s):
                boxes_all.append([box[0]+px, box[1]+py, box[2]+px, box[3]+py])
                scores_all.append(sc)

    if not boxes_all:
        return np.array([]).reshape(0, 4), np.array([])
    bt = torch.tensor(np.array(boxes_all), dtype=torch.float32)
    st = torch.tensor(np.array(scores_all), dtype=torch.float32)
    keep = torch.ops.torchvision.nms(bt, st, iou).numpy()
    return np.array(boxes_all)[keep], np.array(scores_all)[keep]

def fmt_eta(sec):
    if sec < 60: return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s"

# ── Drawing ───────────────────────────────────────────────────────────────────
COLORS = [
    (72, 199, 255), (72, 255, 167), (255, 130, 72), (255, 220, 72),
    (190, 72, 255), (72, 255, 230), (255, 72, 150), (140, 255, 72),
]

def draw(img, boxes, scores, ids=None, trails=None, show_conf=True, show_id=True):
    out = img.copy()
    H, W = out.shape[:2]
    sc = max(0.4, min(1.2, W / 900))

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        tid = int(ids[i]) if ids is not None and i < len(ids) else None
        col = COLORS[tid % len(COLORS)] if tid is not None else COLORS[0]
        cl  = max(8, int((x2-x1)*0.18))
        lw  = max(1, int(2*sc))

        for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out, (px, py), (px+dx*cl, py), col, lw+1, cv2.LINE_AA)
            cv2.line(out, (px, py), (px, py+dy*cl), col, lw+1, cv2.LINE_AA)

        roi = out[y1:y2, x1:x2]
        fill = np.ones_like(roi) * np.array(col, dtype=np.uint8)
        out[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.93, fill, 0.07, 0)

        parts = []
        if show_id and tid: parts.append(f"#{tid}")
        if show_conf: parts.append(f"{score:.0%}")
        label = " ".join(parts)
        if label:
            fs = 0.42 * sc
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ly = max(th + 6, y1)
            roi_bg = out[ly-th-5:ly, x1:x1+tw+6]
            if roi_bg.shape[0] > 0 and roi_bg.shape[1] > 0:
                dark = np.zeros_like(roi_bg)
                out[ly-th-5:ly, x1:x1+tw+6] = cv2.addWeighted(roi_bg, 0.25, dark, 0.75, 0)
            cv2.putText(out, label, (x1+3, ly-4), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2: continue
            col = COLORS[tid % len(COLORS)]
            for j in range(1, len(pts)):
                a = (j / len(pts)) ** 1.5
                cv2.line(out, tuple(map(int, pts[j-1])), tuple(map(int, pts[j])),
                         col, max(1, int(a*2)), cv2.LINE_AA)
    return out

def hud_overlay(img, lines, x=12, y=14, alpha=0.82):
    out = img.copy()
    fs, pad = 0.52, 8
    line_h = 26
    sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs if i else 0.65, 2 if i == 0 else 1)[0]
             for i, l in enumerate(lines)]
    bw = max(s[0] for s in sizes) + pad * 2
    bh = len(lines) * line_h + pad
    roi = out[y:y+bh, x:x+bw]
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        dark = np.zeros_like(roi)
        out[y:y+bh, x:x+bw] = cv2.addWeighted(roi, 1-alpha, dark, alpha, 0)
    cy = y + pad + 18
    for i, line in enumerate(lines):
        col = (72, 255, 167) if i == 0 else (200, 200, 200)
        fs2 = 0.65 if i == 0 else 0.42
        lw  = 2 if i == 0 else 1
        cv2.putText(out, line, (x+pad, cy), cv2.FONT_HERSHEY_SIMPLEX, fs2, col, lw, cv2.LINE_AA)
        cy += line_h if i == 0 else 20
    return out

# ── Auto-detect petri dish ────────────────────────────────────────────────────
def find_dish(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    H, W = frame.shape[:2]
    min_r, max_r = min(H, W)//6, min(H, W)//2
    for dp in [1.2, 1.5, 2.0]:
        for p1 in [80, 50, 30]:
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp, min(H,W)//2,
                                       param1=p1, param2=50, minRadius=min_r, maxRadius=max_r)
            if circles is not None:
                c = np.round(circles[0]).astype(int)
                best = max(c, key=lambda x: x[2])
                return int(best[0]), int(best[1]), int(best[2])
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        if r > min_r:
            return int(cx), int(cy), int(r)
    return None

def dish_crop(cx, cy, r, pad, W, H):
    if r == 0:
        return 0, 0, W, H
    p = int(pad)
    return max(0, cx-r-p), max(0, cy-r-p), min(W, cx+r+p), min(H, cy+r+p)

# ── Tab 1: Image Analysis ─────────────────────────────────────────────────────
def analyze_image(image, conf, iou, quality):
    if image is None:
        return None, "Upload an image to get started.", None
    overlap = QUALITY[quality]["overlap"]
    t0 = time.time()
    boxes, scores = detect(image, conf, iou, overlap)
    elapsed = time.time() - t0
    n = len(boxes)

    out = draw(image, boxes, scores, show_conf=True, show_id=False)
    H, W = out.shape[:2]
    xs, ys = _patch_positions(W, H, PATCH_SIZE, overlap)
    patches = len(xs) * len(ys) if (W > PATCH_SIZE*1.5 or H > PATCH_SIZE*1.5) else 1

    out = hud_overlay(out, [
        f"{n} worms detected",
        f"{elapsed:.1f}s  ·  {patches} patches  ·  {quality}  ·  {HW['device']}",
    ])

    stats = f"**{n} worms** found in {elapsed:.1f}s"
    if n > 0:
        avg_c = np.mean(scores)
        w_px  = boxes[:,2] - boxes[:,0]
        h_px  = boxes[:,3] - boxes[:,1]
        stats += f" · avg confidence {avg_c:.0%} · avg size {np.mean(w_px):.0f}×{np.mean(h_px):.0f}px"

    csv_path = None
    if n > 0:
        csv_path = os.path.join(tempfile.gettempdir(), "worm_detections.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id","x1","y1","x2","y2","conf","width","height","area"])
            for i, (b, s) in enumerate(zip(boxes, scores)):
                x1,y1,x2,y2 = b
                w.writerow([i+1, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                             f"{s:.4f}", f"{x2-x1:.1f}", f"{y2-y1:.1f}", f"{(x2-x1)*(y2-y1):.0f}"])
    return out, stats, csv_path

# ── Tab 2: Video Tracking ─────────────────────────────────────────────────────
class Tracker:
    def __init__(self, max_gone=15, max_dist=80):
        self.nid = 1
        self.objs = {}; self.bxs = {}; self.gone = {}
        self.max_gone = max_gone; self.max_dist = max_dist

    def update(self, dets):
        if len(dets) == 0:
            for oid in list(self.gone):
                self.gone[oid] += 1
                if self.gone[oid] > self.max_gone:
                    del self.objs[oid]; del self.bxs[oid]; del self.gone[oid]
            return []
        cens = np.array([((d[0]+d[2])/2, (d[1]+d[3])/2) for d in dets])
        if not self.objs:
            res = []
            for d, c in zip(dets, cens):
                self.objs[self.nid] = c; self.bxs[self.nid] = d; self.gone[self.nid] = 0
                res.append((self.nid, d)); self.nid += 1
            return res
        from scipy.spatial.distance import cdist
        ids = list(self.objs); oc = np.array([self.objs[i] for i in ids])
        D = cdist(oc, cens)
        rows = D.min(1).argsort(); cols = D.argmin(1)[rows]
        used_r, used_c = set(), set(); res = []
        for r, c in zip(rows, cols):
            if r in used_r or c in used_c or D[r,c] > self.max_dist: continue
            oid = ids[r]
            self.objs[oid] = cens[c]; self.bxs[oid] = dets[c]; self.gone[oid] = 0
            used_r.add(r); used_c.add(c); res.append((oid, dets[c]))
        for c in range(len(cens)):
            if c not in used_c:
                self.objs[self.nid] = cens[c]; self.bxs[self.nid] = dets[c]; self.gone[self.nid] = 0
                res.append((self.nid, dets[c])); self.nid += 1
        for r in range(len(ids)):
            if r not in used_r:
                oid = ids[r]; self.gone[oid] += 1
                if self.gone[oid] > self.max_gone:
                    del self.objs[oid]; del self.bxs[oid]; del self.gone[oid]
        return res

def track_video(video_path, conf, iou, quality, trail_len, cx1, cy1, cx2, cy2,
                trim_s, trim_e, progress=gr.Progress()):
    if video_path is None:
        return None, "No video uploaded.", None, None

    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps

    cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
    use_crop = not (cx1 == 0 and cy1 == 0 and cx2 == 0 and cy2 == 0)
    if use_crop:
        cx1 = max(0, min(cx1, ow-1)); cy1 = max(0, min(cy1, oh-1))
        cx2 = max(cx1+10, min(cx2, ow)); cy2 = max(cy1+10, min(cy2, oh))
    else:
        cx1, cy1, cx2, cy2 = 0, 0, ow, oh
    W, H = cx2-cx1, cy2-cy1

    trim_s = max(0, float(trim_s)); trim_e = float(trim_e)
    if trim_e <= trim_s: trim_e = dur
    sf = int(trim_s * fps); ef = min(int(trim_e * fps), total)
    total_f = ef - sf

    preset = QUALITY[quality]
    voverlap = preset["overlap"]; skip = max(1, preset["skip"])
    to_process = total_f // skip

    out_path = os.path.join(tempfile.gettempdir(), "tracked.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracker = Tracker(max_gone=int(fps*2))
    counts = []; all_tracks = {}; trails = defaultdict(list)
    last = None; processed = 0; t0 = time.time(); eta_val = 999

    if sf > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    progress(0, desc="Starting…")

    for fn in range(total_f):
        ok, frame = cap.read()
        if not ok: break
        if use_crop: frame = frame[cy1:cy2, cx1:cx2]

        if fn % skip == 0:
            boxes, scores = detect(frame, conf, iou, voverlap)
            tracked = tracker.update(boxes)
            tids, tboxes = [], []
            for tid, box in tracked:
                tids.append(tid); tboxes.append(box)
                cx_p = (box[0]+box[2])/2; cy_p = (box[1]+box[3])/2
                if tid not in all_tracks:
                    all_tracks[tid] = {"first": fn, "pos": []}
                all_tracks[tid]["last"] = fn
                all_tracks[tid]["pos"].append((cx_p, cy_p, fn))
                trails[tid].append((cx_p, cy_p))
                if len(trails[tid]) > trail_len: trails[tid] = trails[tid][-trail_len:]
            tboxes = np.array(tboxes) if tboxes else np.array([]).reshape(0,4)
            tscores = scores[:len(tboxes)] if len(scores) >= len(tboxes) else np.ones(len(tboxes))
            counts.append(len(tboxes))
            annotated = draw(frame, tboxes, tscores, np.array(tids) if tids else None, trails, show_conf=False)
            annotated = hud_overlay(annotated, [
                f"{len(tboxes)} worms  ·  {len(all_tracks)} unique",
                f"t={fn/fps:.1f}s  frame {fn}/{total}  ·  ETA {fmt_eta(eta_val)}",
            ])
            last = annotated; processed += 1
            if processed > 1:
                eta_val = (to_process - processed) * (time.time()-t0) / processed
        else:
            annotated = last if last is not None else frame

        writer.write(annotated)
        if fn % 15 == 0:
            progress(fn/total_f, desc=f"Frame {fn}/{total_f} · ETA {fmt_eta(eta_val)}")

    cap.release(); writer.release()

    csv_path = os.path.join(tempfile.gettempdir(), "tracking.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","first_frame","last_frame","duration_s","distance_px","velocity_px_s","n_detections"])
        for tid, d in sorted(all_tracks.items()):
            pos = d["pos"]; first = d["first"]; last_f = d["last"]
            dur_t = (last_f - first) / fps
            dist = sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
            vel = dist / dur_t if dur_t > 0 else 0
            w.writerow([tid, first, last_f, f"{dur_t:.2f}", f"{dist:.1f}", f"{vel:.2f}", len(pos)])

    total_t = time.time() - t0
    n_unique = len(all_tracks)
    avg_ct = np.mean(counts) if counts else 0
    max_ct = max(counts) if counts else 0

    vels, dists = [], []
    for d in all_tracks.values():
        pos = d["pos"]
        if len(pos) < 2: continue
        dist = sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
        dur_t = (d["last"]-d["first"])/fps
        dists.append(dist)
        if dur_t > 0: vels.append(dist/dur_t)

    stats_md = f"""**{n_unique} unique worms tracked** in {fmt_eta(total_t)}

| | |
|---|---|
| Duration | {total_f/fps:.1f}s @ {fps:.0f}fps |
| Avg count/frame | {avg_ct:.1f} |
| Peak count | {max_ct} |
| Avg velocity | {np.mean(vels):.1f} px/s |
| Avg distance | {np.mean(dists):.0f} px |
"""
    analytics = make_analytics(all_tracks, counts, fps, W, H)
    return out_path, stats_md, csv_path, analytics

def make_analytics(all_tracks, counts, fps, W, H):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs
        from scipy.ndimage import gaussian_filter

        BG="#f5f6f8"; PNL="#ffffff"; TXT="#1a1d23"; ACC="#2563eb"; ACC2="#16a34a"; GRD="#e4e7ec"
        fig = plt.figure(figsize=(15, 8), facecolor=BG)
        g = gs.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35, left=0.07, right=0.97, top=0.90, bottom=0.10)
        axes = [fig.add_subplot(g[0,:2]), fig.add_subplot(g[0,2]),
                fig.add_subplot(g[1,:2]), fig.add_subplot(g[1,2])]
        for ax in axes:
            ax.set_facecolor(PNL)
            for sp in ax.spines.values(): sp.set_color(GRD)
            ax.tick_params(colors=TXT, labelsize=8); ax.grid(color=GRD, lw=0.5, alpha=0.8)
            ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT); ax.title.set_color(TXT)
            ax.title.set_fontsize(9); ax.title.set_fontweight("bold")

        ts = [i/fps for i in range(len(counts))]
        axes[0].plot(ts, counts, color=ACC, lw=1.5, alpha=0.9)
        axes[0].fill_between(ts, counts, alpha=0.12, color=ACC)
        if len(counts) > 10:
            ww = max(3, len(counts)//20)
            roll = np.convolve(counts, np.ones(ww)/ww, "valid")
            rt = ts[ww//2:ww//2+len(roll)]
            axes[0].plot(rt, roll, color=ACC2, lw=2, ls="--", alpha=0.8, label="Rolling avg")
            axes[0].legend(facecolor=PNL, edgecolor=GRD, labelcolor=TXT, fontsize=7)
        axes[0].set(xlabel="Time (s)", ylabel="Worm count", title="Count Over Time")

        vels = []
        for d in all_tracks.values():
            pos = d["pos"]
            if len(pos) < 2: continue
            dist = sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
            dur_t = (d["last"]-d["first"])/fps
            if dur_t > 0: vels.append(dist/dur_t)
        if vels:
            axes[1].hist(vels, bins=15, color=ACC, edgecolor=BG, alpha=0.85)
            axes[1].axvline(np.mean(vels), color=ACC2, lw=1.5, ls="--", label=f"μ={np.mean(vels):.0f}")
            axes[1].legend(facecolor=PNL, edgecolor=GRD, labelcolor=TXT, fontsize=7)
        axes[1].set(xlabel="Velocity (px/s)", ylabel="Count", title="Velocity Distribution")

        clrs = plt.cm.tab10(np.linspace(0, 1, max(1, len(all_tracks))))
        for idx, (tid, d) in enumerate(all_tracks.items()):
            pos = d["pos"]
            if len(pos) < 2: continue
            xs2 = [p[0] for p in pos]; ys2 = [p[1] for p in pos]
            axes[2].plot(xs2, ys2, color=clrs[idx%len(clrs)], lw=0.8, alpha=0.7)
            axes[2].scatter(xs2[0], ys2[0], color=ACC2, s=12, zorder=5)
            axes[2].scatter(xs2[-1], ys2[-1], color="#dc2626", s=12, marker="x", lw=1.5, zorder=5)
        axes[2].set_xlim(0, W); axes[2].set_ylim(H, 0)
        axes[2].set(xlabel="X (px)", ylabel="Y (px)", title="Movement Paths  ● start  × end")
        axes[2].set_aspect("equal", "datalim")

        sc = 8
        hm = np.zeros((H//sc+1, W//sc+1), dtype=np.float32)
        for d in all_tracks.values():
            for x, y, _ in d["pos"]:
                hx = min(int(x/sc), hm.shape[1]-1); hy = min(int(y/sc), hm.shape[0]-1)
                hm[hy, hx] += 1
        hm = gaussian_filter(hm, sigma=3)
        im = axes[3].imshow(hm, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=axes[3], shrink=0.85).ax.tick_params(labelcolor=TXT, labelsize=7)
        axes[3].set(title="Activity Heatmap", xlabel="X", ylabel="Y")

        fig.suptitle("Worm Tracking Analytics", fontsize=13, color=TXT, fontweight="bold", y=0.97)
        out = os.path.join(tempfile.gettempdir(), "analytics.png")
        plt.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight"); plt.close()
        img = cv2.imread(out)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Analytics error: {e}"); return None

# ── Tab 3: Batch ──────────────────────────────────────────────────────────────
def batch_process(files, conf, iou, quality, progress=gr.Progress()):
    if not files:
        return None, "Upload some images first.", None
    overlap = QUALITY[quality]["overlap"]
    out_dir = os.path.join(tempfile.gettempdir(), "worm_batch")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    stats_rows = []; gallery = []
    for i, fp in enumerate(files):
        progress((i+1)/len(files), desc=f"Image {i+1}/{len(files)}")
        img = cv2.imread(fp)
        if img is None: continue
        fname = os.path.basename(fp)
        boxes, scores = detect(img, conf, iou, overlap)
        ann = draw(img, boxes, scores, show_conf=True, show_id=False)
        cv2.imwrite(os.path.join(out_dir, f"detected_{fname}"), ann)
        gallery.append((cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), f"{fname}: {len(boxes)} worms"))
        avg_c = float(np.mean(scores)) if len(scores) else 0
        stats_rows.append({"file": fname, "count": len(boxes), "conf": f"{avg_c:.0%}"})

    csv_p = os.path.join(out_dir, "summary.csv")
    with open(csv_p, "w", newline="") as f:
        dw = csv.DictWriter(f, ["file","count","conf"]); dw.writeheader(); dw.writerows(stats_rows)

    zip_p = shutil.make_archive(os.path.join(tempfile.gettempdir(), "worm_batch"), "zip", out_dir)
    total = sum(r["count"] for r in stats_rows)
    avg   = total / len(stats_rows) if stats_rows else 0

    md = f"**{len(stats_rows)} images processed** · {total} total worms · {avg:.1f} avg\n\n"
    md += "| File | Worms | Confidence |\n|---|---|---|\n"
    for r in stats_rows:
        md += f"| {r['file']} | {r['count']} | {r['conf']} |\n"
    return gallery, md, zip_p

# ── Tab 4: Frame Explorer ─────────────────────────────────────────────────────
def analyze_frame(video_path, conf, frame_no, cx1, cy1, cx2, cy2):
    if not video_path: return None, "Upload a video first."
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fn    = min(max(0, int(frame_no)), total-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn); ok, frame = cap.read(); cap.release()
    if not ok: return None, "Cannot read that frame."

    cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)
    if not (cx1 == 0 and cy1 == 0 and cx2 == 0 and cy2 == 0):
        cx1 = max(0,min(cx1,ow-1)); cy1 = max(0,min(cy1,oh-1))
        cx2 = max(cx1+10,min(cx2,ow)); cy2 = max(cy1+10,min(cy2,oh))
        frame = frame[cy1:cy2, cx1:cx2]

    boxes, scores = detect(frame, conf)
    ann = draw(frame, boxes, scores, show_conf=True, show_id=False)
    ann = hud_overlay(ann, [f"{len(boxes)} worms", f"frame {fn}  ·  t={fn/fps:.2f}s"])
    return cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), \
           (f"**Frame {fn}** (t={fn/fps:.2f}s) — **{len(boxes)} worms** · avg conf {np.mean(scores):.0%}"
            if len(scores) else f"**Frame {fn}** — 0 worms")

# ── Shared video helpers ──────────────────────────────────────────────────────
def _load_video_meta(path):
    if path is None:
        return None, None, 0, 0, 0, 0, "", 0
    cap = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read(); cap.release()
    dur = total / fps
    if not ok:
        return None, None, 0, 0, W, H, "Cannot read video.", dur

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dish = find_dish(frame)
    if dish:
        cx, cy, r = dish
        x1, y1, x2, y2 = dish_crop(cx, cy, r, 50, W, H)
        info = f"{W}×{H}, {dur:.1f}s · Dish detected ✓"
        dd = {"cx": cx, "cy": cy, "r": r, "W": W, "H": H}
    else:
        x1, y1, x2, y2 = 0, 0, W, H
        info = f"{W}×{H}, {dur:.1f}s · No dish detected"
        dd = {"cx": 0, "cy": 0, "r": 0, "W": W, "H": H}

    return dd, frame_rgb, x1, y1, x2, y2, info, round(dur, 1)

def _make_crop_preview(dd, frame_rgb, x1, y1, x2, y2):
    if frame_rgb is None or dd is None:
        return None
    preview = frame_rgb.copy()
    W, H = dd["W"], dd["H"]
    if dd["r"] > 0:
        cv2.circle(preview, (dd["cx"], dd["cy"]), dd["r"], (37, 99, 235), 2)
    if not (x1 == 0 and y1 == 0 and x2 == W and y2 == H):
        darkened = (preview * 0.35).astype(np.uint8)
        mask = np.zeros_like(preview, dtype=np.uint8)
        cv2.rectangle(mask, (x1,y1), (x2,y2), (255,255,255), -1)
        preview = np.where(mask > 0, preview, darkened)
        cv2.rectangle(preview, (x1,y1), (x2,y2), (37, 99, 235), 3)
        cv2.putText(preview, f"{x2-x1}x{y2-y1}px", (x1+6, y1+26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (37,99,235), 2, cv2.LINE_AA)
    return preview

def _update_crop(dd, frame_rgb, auto, pad):
    if dd is None: return 0, 0, 0, 0, None
    W, H = dd["W"], dd["H"]
    if auto:
        x1, y1, x2, y2 = dish_crop(dd["cx"], dd["cy"], dd["r"], pad, W, H)
    else:
        x1, y1, x2, y2 = 0, 0, W, H
    return x1, y1, x2, y2, _make_crop_preview(dd, frame_rgb, x1, y1, x2, y2)

# ── Video crop/trim ───────────────────────────────────────────────────────────
def crop_trim(path, x1, y1, x2, y2, ts, te, progress=gr.Progress()):
    if path is None: return None, "No video.", None
    cap = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    x1=max(0,min(x1,ow-1)); y1=max(0,min(y1,oh-1))
    x2=max(x1+10,min(x2,ow)); y2=max(y1+10,min(y2,oh))
    ts=max(0,float(ts)); te=float(te)
    if te<=ts: te=dur
    sf=int(ts*fps); ef=int(te*fps); nf=ef-sf
    CW,CH = x2-x1, y2-y1
    out_p = os.path.join(tempfile.gettempdir(), "cropped.mp4")
    wt = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (CW, CH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    t0 = time.time(); written = 0
    for i in range(nf):
        ok, frame = cap.read()
        if not ok: break
        wt.write(frame[y1:y2, x1:x2]); written += 1
        if written % 100 == 0:
            el = time.time()-t0; rate = written/el if el>0 else 1; rem = (nf-written)/rate
            progress(written/nf, desc=f"Frame {written}/{nf} · ETA {fmt_eta(rem)}")
    cap.release(); wt.release()
    t_total = time.time()-t0
    speedup = (ow*oh) / (CW*CH) if CW*CH else 1
    md = f"**Done** · {CW}×{CH}px, {te-ts:.1f}s · {t_total:.1f}s to process · ~{speedup:.1f}× detection speedup"
    return out_p, md, out_p

# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
    --bg:      #f4f5f7;
    --surface: #ffffff;
    --border:  #e1e4ea;
    --text:    #111827;
    --muted:   #6b7280;
    --accent:  #2563eb;
    --accent-h:#1d4ed8;
    --green:   #16a34a;
    --r:       10px;
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.app-bar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 24px;
    height: 54px;
    position: sticky;
    top: 0;
    z-index: 100;
}
.app-bar-logo {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.app-bar h1 {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin: 0 !important;
}
.app-bar-tag {
    font-size: 10px; font-weight: 600;
    color: var(--accent); background: #eff6ff;
    border: 1px solid #bfdbfe; padding: 2px 7px; border-radius: 99px;
}
.app-bar-hw { margin-left: auto; font-size: 11.5px; color: var(--muted); }

.tab-nav {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 20px !important;
}
.tab-nav button {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 13px 15px !important; border-radius: 0 !important;
    transition: color .15s !important;
}
.tab-nav button:hover { color: var(--text) !important; }
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

.gradio-tabitem, div[class*="tabitem"] { padding: 20px 24px !important; }

.block, .form, .panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}

label > span, .label-wrap > span {
    font-size: 12px !important; font-weight: 500 !important;
    color: var(--muted) !important; text-transform: none !important;
    letter-spacing: 0 !important; font-family: 'Inter', sans-serif !important;
}

input[type=number], input[type=text], textarea {
    background: var(--bg) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--text) !important;
    font-size: 13px !important; font-family: 'Inter', sans-serif !important;
}
input[type=number]:focus, input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important; outline: none !important;
}
input[type=range] { accent-color: var(--accent) !important; }
input[type=checkbox], input[type=radio] { accent-color: var(--accent) !important; }

button.primary {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: background .15s, transform .1s !important;
}
button.primary:hover { background: var(--accent-h) !important; transform: translateY(-1px) !important; }
button.secondary {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 500 !important;
}
button.secondary:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

.accordion { border: 1px solid var(--border) !important; border-radius: 7px !important; background: var(--bg) !important; }
.accordion .label-wrap { padding: 10px 14px !important; }

.prose, .md { font-size: 13px !important; line-height: 1.65 !important; color: var(--text) !important; }
.prose h2, .md h2 { font-size: 15px !important; font-weight: 600 !important; margin-bottom: 10px !important; color: var(--text) !important; }
.prose strong, .md strong { color: var(--green) !important; font-weight: 600 !important; }
.prose table, .md table { width: 100% !important; border-collapse: collapse !important; font-size: 12.5px !important; margin-top: 8px !important; }
.prose th, .md th { background: var(--bg) !important; padding: 6px 10px !important; font-size: 11px !important; font-weight: 600 !important; color: var(--muted) !important; text-align: left !important; }
.prose td, .md td { padding: 6px 10px !important; border-bottom: 1px solid var(--border) !important; }
.prose tr:last-child td, .md tr:last-child td { border: none !important; }

.image-container img { border-radius: 8px !important; border: 1px solid var(--border) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

footer { display: none !important; }
.footer { padding: 10px 24px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); background: var(--surface); }
"""

with gr.Blocks(title="Worm Analyzer", css=CSS) as app:

    gr.HTML(f"""
    <div class="app-bar">
        <div class="app-bar-logo">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round">
                <path d="M3 12c0-3 2-5 4-5s3 2 5 2 3-2 5-2 4 2 4 5-2 5-4 5-3-2-5-2-3 2-5 2-4-2-4-5z"/>
            </svg>
        </div>
        <h1>Worm Analyzer</h1>
        <span class="app-bar-tag">YOLOv8m</span>
        <span class="app-bar-hw">{HW['device']} · {HW['name'][:30]} · {HW['ms']:.0f}ms/patch</span>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Image Analysis ─────────────────────────────────────────────
        with gr.Tab("Image Analysis"):
            with gr.Row(equal_height=False, variant="panel"):
                with gr.Column(scale=1, min_width=260):
                    img_in   = gr.Image(type="numpy", label="Image", height=280)
                    img_q    = gr.Radio(["Fast","Balanced","Quality"], value=DEFAULT_Q, label="Quality")
                    with gr.Row():
                        img_conf = gr.Slider(.1,.95, DEFAULT_CONF, step=.05, label="Confidence")
                        img_iou  = gr.Slider(.1,.9,  DEFAULT_IOU,  step=.05, label="NMS IoU")
                    img_btn  = gr.Button("Analyze", variant="primary", size="lg")
                with gr.Column(scale=2):
                    img_out  = gr.Image(label="Result", height=360)
                    img_stat = gr.Markdown("Upload an image and click **Analyze**.")
                    img_csv  = gr.File(label="Download CSV")

            img_btn.click(analyze_image, [img_in, img_conf, img_iou, img_q], [img_out, img_stat, img_csv])

        # ── Tab 2: Video Tracking ─────────────────────────────────────────────
        with gr.Tab("Video Tracking"):
            with gr.Row(equal_height=False, variant="panel"):
                with gr.Column(scale=1, min_width=280):
                    vt_vid  = gr.Video(label="Video", height=220)
                    vt_info = gr.Markdown("Upload a video to begin.")
                    with gr.Accordion("Crop region", open=True):
                        with gr.Row():
                            vt_auto = gr.Checkbox(label="Auto-crop dish", value=True)
                            vt_pad  = gr.Slider(-50,150,50,step=5,label="Padding (px)")
                        vt_prev = gr.Image(label=None, height=160, show_label=False)
                        vt_x1 = gr.Number(0,visible=False,precision=0)
                        vt_y1 = gr.Number(0,visible=False,precision=0)
                        vt_x2 = gr.Number(0,visible=False,precision=0)
                        vt_y2 = gr.Number(0,visible=False,precision=0)
                    vt_q    = gr.Radio(["Fast","Balanced","Quality"], value=DEFAULT_Q, label="Quality")
                    with gr.Row():
                        vt_conf  = gr.Slider(.1,.95,DEFAULT_CONF,step=.05,label="Confidence")
                        vt_iou   = gr.Slider(.1,.9, DEFAULT_IOU, step=.05,label="NMS IoU")
                    vt_trail = gr.Slider(10,200,60,step=10,label="Trail length")
                    with gr.Row():
                        vt_ts = gr.Number(0,label="Trim start (s)",precision=1)
                        vt_te = gr.Number(0,label="Trim end (s, 0=full)",precision=1)
                    vt_btn  = gr.Button("Start Tracking", variant="primary", size="lg")
                with gr.Column(scale=2):
                    vt_out   = gr.Video(label="Tracked Video", height=360)
                    vt_stat  = gr.Markdown("*Run tracking to see stats.*")
                    vt_anal  = gr.Image(label=None, height=240, show_label=False)
                    vt_csv   = gr.File(label="Download CSV")

            vt_dd  = gr.State()
            vt_frm = gr.State()

            def _vt_upload(p):
                dd, frm, x1, y1, x2, y2, info, dur = _load_video_meta(p)
                prev = _make_crop_preview(dd, frm, x1, y1, x2, y2) if dd else None
                return dd, frm, x1, y1, x2, y2, info, prev, 0, dur

            def _vt_crop(dd, frm, auto, pad):
                x1, y1, x2, y2, prev = _update_crop(dd, frm, auto, pad)
                return x1, y1, x2, y2, prev

            vt_vid.change(_vt_upload, [vt_vid],
                          [vt_dd,vt_frm,vt_x1,vt_y1,vt_x2,vt_y2,vt_info,vt_prev,vt_ts,vt_te])
            vt_auto.change(_vt_crop, [vt_dd,vt_frm,vt_auto,vt_pad], [vt_x1,vt_y1,vt_x2,vt_y2,vt_prev])
            vt_pad.change( _vt_crop, [vt_dd,vt_frm,vt_auto,vt_pad], [vt_x1,vt_y1,vt_x2,vt_y2,vt_prev])
            vt_btn.click(track_video,
                         [vt_vid,vt_conf,vt_iou,vt_q,vt_trail,vt_x1,vt_y1,vt_x2,vt_y2,vt_ts,vt_te],
                         [vt_out,vt_stat,vt_csv,vt_anal])

        # ── Tab 3: Batch Processing ───────────────────────────────────────────
        with gr.Tab("Batch Processing"):
            with gr.Row(equal_height=False, variant="panel"):
                with gr.Column(scale=1, min_width=260):
                    bt_files = gr.File(file_count="multiple", file_types=["image"], label="Images")
                    bt_q     = gr.Radio(["Fast","Balanced","Quality"], value=DEFAULT_Q, label="Quality")
                    with gr.Row():
                        bt_conf = gr.Slider(.1,.95,DEFAULT_CONF,step=.05,label="Confidence")
                        bt_iou  = gr.Slider(.1,.9, DEFAULT_IOU, step=.05,label="NMS IoU")
                    bt_btn  = gr.Button("Process All", variant="primary", size="lg")
                    bt_stat = gr.Markdown("Upload images and click **Process All**.")
                    bt_zip  = gr.File(label="Download ZIP")
                with gr.Column(scale=3):
                    bt_gallery = gr.Gallery(label="Results", columns=4, height=500)

            bt_btn.click(batch_process, [bt_files,bt_conf,bt_iou,bt_q], [bt_gallery,bt_stat,bt_zip])

        # ── Tab 4: Frame Explorer ─────────────────────────────────────────────
        with gr.Tab("Frame Explorer"):
            with gr.Row(equal_height=False, variant="panel"):
                with gr.Column(scale=1, min_width=260):
                    fe_vid  = gr.Video(label="Video", height=200)
                    fe_frm  = gr.Slider(0,1000,0,step=1,label="Frame")
                    fe_conf = gr.Slider(.1,.95,DEFAULT_CONF,step=.05,label="Confidence")
                    with gr.Accordion("Crop", open=False):
                        fe_msg = gr.Markdown("*Upload video for auto-crop.*")
                        with gr.Row():
                            fe_auto = gr.Checkbox(label="Auto-crop",value=True)
                            fe_pad  = gr.Slider(-50,150,50,step=5,label="Padding")
                        fe_x1 = gr.Number(0,visible=False,precision=0)
                        fe_y1 = gr.Number(0,visible=False,precision=0)
                        fe_x2 = gr.Number(0,visible=False,precision=0)
                        fe_y2 = gr.Number(0,visible=False,precision=0)
                    fe_btn  = gr.Button("Analyze Frame", variant="primary", size="lg")
                with gr.Column(scale=2):
                    fe_out  = gr.Image(label="Result", height=460)
                    fe_stat = gr.Markdown("*Select a frame and click Analyze.*")

            fe_dd  = gr.State()
            fe_fst = gr.State()

            def _fe_upload(p):
                dd, frm, x1, y1, x2, y2, info, _ = _load_video_meta(p)
                cap = cv2.VideoCapture(p) if p else None
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 1
                if cap: cap.release()
                return gr.Slider(maximum=max(1,total-1)), dd, frm, info, x1, y1, x2, y2

            def _fe_crop(dd, frm, auto, pad):
                x1, y1, x2, y2, _ = _update_crop(dd, frm, auto, pad)
                return x1, y1, x2, y2

            fe_vid.change(_fe_upload, [fe_vid], [fe_frm,fe_dd,fe_fst,fe_msg,fe_x1,fe_y1,fe_x2,fe_y2])
            fe_auto.change(_fe_crop, [fe_dd,fe_fst,fe_auto,fe_pad], [fe_x1,fe_y1,fe_x2,fe_y2])
            fe_pad.change( _fe_crop, [fe_dd,fe_fst,fe_auto,fe_pad], [fe_x1,fe_y1,fe_x2,fe_y2])
            fe_btn.click(analyze_frame, [fe_vid,fe_conf,fe_frm,fe_x1,fe_y1,fe_x2,fe_y2], [fe_out,fe_stat])

        # ── Tab 5: Video Tools ────────────────────────────────────────────────
        with gr.Tab("Video Tools"):
            with gr.Row(equal_height=False, variant="panel"):
                with gr.Column(scale=1, min_width=260):
                    vl_vid  = gr.Video(label="Video", height=200)
                    vl_info = gr.Markdown("*Upload a video.*")
                    with gr.Accordion("Crop", open=True):
                        with gr.Row():
                            vl_auto = gr.Checkbox(label="Auto-crop dish",value=True)
                            vl_pad  = gr.Slider(-50,150,50,step=5,label="Padding")
                        vl_x1 = gr.Number(0,visible=False,precision=0)
                        vl_y1 = gr.Number(0,visible=False,precision=0)
                        vl_x2 = gr.Number(1920,visible=False,precision=0)
                        vl_y2 = gr.Number(1080,visible=False,precision=0)
                    with gr.Row():
                        vl_ts = gr.Number(0,label="Trim start (s)",precision=1)
                        vl_te = gr.Number(0,label="Trim end (s, 0=full)",precision=1)
                    vl_btn  = gr.Button("Crop & Trim", variant="primary", size="lg")
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("Preview"):
                            vl_prev = gr.Image(label=None, height=360, show_label=False)
                        with gr.Tab("Result"):
                            vl_out  = gr.Video(label=None, height=360, show_label=False)
                    vl_stat = gr.Markdown("*Process a video to see stats.*")
                    vl_dl   = gr.File(label="Download")

            vl_dd  = gr.State()
            vl_frm = gr.State()

            def _vl_upload(p):
                dd, frm, x1, y1, x2, y2, info, dur = _load_video_meta(p)
                prev = _make_crop_preview(dd, frm, x1, y1, x2, y2) if dd else None
                return dd, frm, info, x1, y1, x2, y2, 0, dur, prev

            def _vl_crop(dd, frm, auto, pad):
                x1, y1, x2, y2, prev = _update_crop(dd, frm, auto, pad)
                return x1, y1, x2, y2, prev

            vl_vid.change(_vl_upload, [vl_vid],
                          [vl_dd,vl_frm,vl_info,vl_x1,vl_y1,vl_x2,vl_y2,vl_ts,vl_te,vl_prev])
            vl_auto.change(_vl_crop, [vl_dd,vl_frm,vl_auto,vl_pad], [vl_x1,vl_y1,vl_x2,vl_y2,vl_prev])
            vl_pad.change( _vl_crop, [vl_dd,vl_frm,vl_auto,vl_pad], [vl_x1,vl_y1,vl_x2,vl_y2,vl_prev])
            vl_btn.click(crop_trim, [vl_vid,vl_x1,vl_y1,vl_x2,vl_y2,vl_ts,vl_te],
                         [vl_out,vl_stat,vl_dl])

        # ── Tab 6: Help ───────────────────────────────────────────────────────
        with gr.Tab("Help"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
## How to use

**Image Analysis** — Upload a photo, pick a quality preset, click Analyze. Download the CSV for coordinates and sizes.

**Video Tracking** — Upload a video. The app auto-finds the petri dish and suggests a crop region. Adjust padding if needed, pick quality, click Start Tracking.

**Batch Processing** — Upload many images at once. Download a ZIP with all annotated images and a CSV summary.

**Frame Explorer** — Scrub through a video one frame at a time. Useful for tuning confidence settings before a full tracking run.

**Video Tools** — Crop and trim a video first. A smaller cropped video processes much faster (speedup shown after processing).

---
## Quality presets

| Preset | Patch overlap | Frame skip | Best for |
|--------|---|---|---|
| Fast | 25% | every 6th | Quick preview |
| Balanced | 40% | every 3rd | Daily use |
| Quality | 50% | every frame | Max accuracy |
""")
                with gr.Column():
                    gr.Markdown(f"""
## System

| | |
|---|---|
| Device | {HW['device']} ({HW['name']}) |
| Speed | {HW['ms']:.0f}ms per 416×416 patch |
| Default quality | {DEFAULT_Q} |
| Model | YOLOv8m · 25.8M parameters |
| Batch inference | {'8 patches/call (GPU)' if HW['device'] in ('CUDA','MPS') else '1 patch/call (CPU)'} |

## Tips
- **Crop first** — less background = more accurate, much faster
- **Confidence 0.3–0.4** for small/dim worms
- **Confidence 0.5–0.6** for clear high-contrast worms
- Use **Video Tools** to pre-crop a long video, then track
""")

    gr.HTML(f'<div class="footer">{HW["device"]} · {HW["name"]} · {HW["ms"]:.0f}ms/patch · Worm Analyzer</div>')


if __name__ == "__main__":
    print("\n  Worm Analyzer  →  http://localhost:7860\n")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)