#!/usr/bin/env python3
"""
Worm Analyzer  ·  Detection, Tracking & Analytics
Run:  python3 worm_analyzer.py   →   http://localhost:7860
"""

import os, sys, cv2, csv, time, shutil, tempfile, threading
import numpy as np
from collections import defaultdict

import gradio as gr
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_worm_yolov8m.pt")
PATCH       = 416
CONF_DEF    = 0.40
IOU_DEF     = 0.50
SPEED_OPTS  = {
    "Fast — preview only":    {"overlap": 0.25, "skip": 6},
    "Normal — recommended":   {"overlap": 0.40, "skip": 3},
    "Thorough — best results":{"overlap": 0.50, "skip": 1},
}

# ─────────────────────────────────────────────────────────
#  MODEL LOAD + HARDWARE
# ─────────────────────────────────────────────────────────
print("Loading model…")
if not os.path.exists(MODEL_PATH):
    sys.exit(f"ERROR: model not found at {MODEL_PATH}")
model = YOLO(MODEL_PATH)

import platform, torch as _torch

def _hw():
    d = {"os": platform.system(), "arch": platform.machine()}
    if _torch.cuda.is_available():
        d["dev"]  = "CUDA GPU"
        d["name"] = _torch.cuda.get_device_name(0)
    elif getattr(getattr(_torch, "backends", None), "mps", None) and _torch.backends.mps.is_available():
        d["dev"]  = "Apple GPU"
        d["name"] = "Apple Silicon"
    else:
        d["dev"]  = "CPU"
        d["name"] = platform.processor() or "CPU"
    dummy = np.random.randint(0, 255, (PATCH, PATCH, 3), dtype=np.uint8)
    for _ in range(2): model(dummy, conf=0.5, imgsz=PATCH, verbose=False)
    t = []
    for _ in range(3):
        t0 = time.time(); model(dummy, conf=0.5, imgsz=PATCH, verbose=False); t.append(time.time()-t0)
    d["ms"] = np.median(t) * 1000
    return d

print("Benchmarking hardware…")
HW = _hw()
DEF_SPEED = ("Thorough — best results" if HW["ms"] < 80 else
             "Normal — recommended"    if HW["ms"] < 250 else
             "Fast — preview only")
print(f"  {HW['dev']} · {HW['ms']:.0f}ms/patch · default={DEF_SPEED}")

import torch

# ─────────────────────────────────────────────────────────
#  DETECTION ENGINE
# ─────────────────────────────────────────────────────────
def _steps(dim, patch, overlap):
    stride = int(patch * (1 - overlap))
    s = list(range(0, max(1, dim - patch + 1), stride))
    if not s or s[-1] + patch < dim: s.append(max(0, dim - patch))
    return s

def run_detection(image, conf=CONF_DEF, iou=IOU_DEF, overlap=0.40):
    H, W = image.shape[:2]
    # Small image → single inference
    if W <= PATCH * 1.5 and H <= PATCH * 1.5:
        r = model(image, conf=conf, iou=iou, imgsz=PATCH, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    # Sliding window
    xs, ys = _steps(W, PATCH, overlap), _steps(H, PATCH, overlap)
    patches, positions = [], []
    for y in ys:
        for x in xs:
            patches.append(image[y:y+PATCH, x:x+PATCH])
            positions.append((x, y))

    batch_size = 8 if "GPU" in HW["dev"] else 1
    boxes_all, scores_all = [], []
    for i in range(0, len(patches), batch_size):
        results = model(patches[i:i+batch_size], conf=conf, iou=iou, imgsz=PATCH, verbose=False)
        for r, (px, py) in zip(results, positions[i:i+batch_size]):
            for box, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                boxes_all.append([box[0]+px, box[1]+py, box[2]+px, box[3]+py])
                scores_all.append(sc)

    if not boxes_all:
        return np.array([]).reshape(0, 4), np.array([])

    bt = torch.tensor(np.array(boxes_all), dtype=torch.float32)
    st = torch.tensor(np.array(scores_all), dtype=torch.float32)
    keep = torch.ops.torchvision.nms(bt, st, iou).numpy()
    return np.array(boxes_all)[keep], np.array(scores_all)[keep]

def fmt_time(s):
    if s < 60: return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"

# ─────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────
PALETTE = [
    (52,  199, 255), (52,  255, 158), (255, 124, 52 ),
    (255, 214, 52 ), (180, 52,  255), (52,  255, 224),
    (255, 52,  140), (130, 255, 52 ), (255, 180, 52 ),
    (52,  130, 255),
]

def annotate(img, boxes, scores, ids=None, trails=None):
    out = img.copy()
    H, W = out.shape[:2]
    scale = max(0.38, min(1.1, W / 1000))

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        tid = int(ids[i]) if ids is not None and i < len(ids) else None
        col = PALETTE[tid % len(PALETTE)] if tid is not None else PALETTE[0]
        cl  = max(7, int((x2-x1) * 0.2))
        lw  = max(1, int(2 * scale))

        # Corner-bracket style (not full rectangle — cleaner)
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out,(px,py),(px+dx*cl,py),col,lw+1,cv2.LINE_AA)
            cv2.line(out,(px,py),(px,py+dy*cl),col,lw+1,cv2.LINE_AA)

        # Tinted fill
        roi  = out[y1:y2, x1:x2]
        fill = np.full_like(roi, col, dtype=np.uint8)
        out[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.92, fill, 0.08, 0)

        # Label
        label = f"#{tid}" if tid else f"{score:.0%}"
        fs = 0.4 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ly = max(th + 5, y1)
        bg = out[ly-th-4:ly, x1:x1+tw+5]
        if bg.shape[0] > 0 and bg.shape[1] > 0:
            out[ly-th-4:ly, x1:x1+tw+5] = cv2.addWeighted(bg, 0.2, np.zeros_like(bg), 0.8, 0)
        cv2.putText(out, label, (x1+2, ly-3), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

    # Trails
    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2: continue
            col = PALETTE[tid % len(PALETTE)]
            for j in range(1, len(pts)):
                a = (j / len(pts)) ** 1.5
                cv2.line(out, tuple(map(int,pts[j-1])), tuple(map(int,pts[j])),
                         col, max(1, int(a*2)), cv2.LINE_AA)
    return out

def stamp(img, big, small):
    """Dark semi-transparent info stamp on image."""
    out = img.copy()
    fs_big, fs_small = 0.65, 0.40
    pad = 10
    (w1,h1),_ = cv2.getTextSize(big,   cv2.FONT_HERSHEY_SIMPLEX, fs_big,   2)
    (w2,h2),_ = cv2.getTextSize(small, cv2.FONT_HERSHEY_SIMPLEX, fs_small, 1)
    bw = max(w1,w2) + pad*2; bh = h1+h2+pad*2+6
    roi = out[8:8+bh, 8:8+bw]
    if roi.shape[0]>0 and roi.shape[1]>0:
        out[8:8+bh, 8:8+bw] = cv2.addWeighted(roi,0.18,np.zeros_like(roi),0.82,0)
    cv2.putText(out, big,   (8+pad, 8+pad+h1),       cv2.FONT_HERSHEY_SIMPLEX, fs_big,   (72,255,158), 2, cv2.LINE_AA)
    cv2.putText(out, small, (8+pad, 8+pad+h1+6+h2),  cv2.FONT_HERSHEY_SIMPLEX, fs_small, (200,200,200), 1, cv2.LINE_AA)
    return out

# ─────────────────────────────────────────────────────────
#  DISH DETECTION
# ─────────────────────────────────────────────────────────
def find_dish(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(gray,(15,15),0)
    H,W  = frame.shape[:2]
    minr, maxr = min(H,W)//6, min(H,W)//2
    for dp in [1.2,1.5,2.0]:
        for p1 in [80,50,30]:
            c = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp,min(H,W)//2,
                                 param1=p1,param2=50,minRadius=minr,maxRadius=maxr)
            if c is not None:
                best = max(np.round(c[0]).astype(int), key=lambda x:x[2])
                return int(best[0]),int(best[1]),int(best[2])
    _,thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (cx,cy),r = cv2.minEnclosingCircle(max(cnts,key=cv2.contourArea))
        if r>minr: return int(cx),int(cy),int(r)
    return None

def dish_to_crop(cx,cy,r,pad,W,H):
    if r==0: return 0,0,W,H
    p = int(pad)
    return max(0,cx-r-p), max(0,cy-r-p), min(W,cx+r+p), min(H,cy+r+p)

# ─────────────────────────────────────────────────────────
#  SHARED VIDEO HELPERS
# ─────────────────────────────────────────────────────────
def load_video(path):
    """Returns (dish_state, frame_rgb, x1,y1,x2,y2, info_text, duration_sec)"""
    if not path: return None,None,0,0,0,0,"",0
    cap = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read(); cap.release()
    dur = total / fps
    if not ok: return None,None,0,0,W,H,"Cannot read video",dur

    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dish = find_dish(frame)
    if dish:
        cx,cy,r = dish
        x1,y1,x2,y2 = dish_to_crop(cx,cy,r,50,W,H)
        dd   = {"cx":cx,"cy":cy,"r":r,"W":W,"H":H}
        info = f"{W}×{H} · {dur:.1f}s · Dish found ✓"
    else:
        x1,y1,x2,y2 = 0,0,W,H
        dd   = {"cx":0,"cy":0,"r":0,"W":W,"H":H}
        info = f"{W}×{H} · {dur:.1f}s · No dish found — full frame will be used"
    return dd, rgb, x1, y1, x2, y2, info, round(dur,1)

def make_preview(dd, rgb, x1,y1,x2,y2):
    if rgb is None or dd is None: return None
    prev = rgb.copy()
    W,H  = dd["W"],dd["H"]
    if dd["r"]>0:
        cv2.circle(prev,(dd["cx"],dd["cy"]),dd["r"],(37,99,235),2)
    if not (x1==0 and y1==0 and x2==W and y2==H):
        dark = (prev*0.3).astype(np.uint8)
        mask = np.zeros_like(prev); cv2.rectangle(mask,(x1,y1),(x2,y2),(255,255,255),-1)
        prev = np.where(mask>0, prev, dark)
        cv2.rectangle(prev,(x1,y1),(x2,y2),(37,99,235),3)
        label = f"Scan area: {x2-x1}×{y2-y1}px"
        cv2.putText(prev,label,(x1+8,y1+28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(37,99,235),2,cv2.LINE_AA)
    return prev

def update_crop(dd,rgb,auto,pad):
    if not dd: return 0,0,0,0,None
    W,H = dd["W"],dd["H"]
    x1,y1,x2,y2 = dish_to_crop(dd["cx"],dd["cy"],dd["r"],pad,W,H) if auto else (0,0,W,H)
    return x1,y1,x2,y2, make_preview(dd,rgb,x1,y1,x2,y2)

# ─────────────────────────────────────────────────────────
#  ANALYTICS CHART
# ─────────────────────────────────────────────────────────
def make_chart(all_tracks, counts, fps, W, H):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.gridspec as mgs
        from scipy.ndimage import gaussian_filter

        BG="#f8f9fa"; PNL="#ffffff"; TXT="#1a1d23"; A1="#2563eb"; A2="#16a34a"; GRD="#e9ecef"
        fig = plt.figure(figsize=(14,7), facecolor=BG)
        g   = mgs.GridSpec(2,3,figure=fig,hspace=0.52,wspace=0.38,left=0.07,right=0.97,top=0.91,bottom=0.10)
        axes= [fig.add_subplot(g[0,:2]),fig.add_subplot(g[0,2]),
               fig.add_subplot(g[1,:2]),fig.add_subplot(g[1,2])]

        for ax in axes:
            ax.set_facecolor(PNL)
            for sp in ax.spines.values(): sp.set_color(GRD)
            ax.tick_params(colors=TXT,labelsize=8)
            ax.grid(color=GRD,lw=0.6); ax.xaxis.label.set_color(TXT)
            ax.yaxis.label.set_color(TXT); ax.title.set_color(TXT)
            ax.title.set_fontsize(9); ax.title.set_fontweight("bold")

        # Count over time
        ts = [i/max(fps,1) for i in range(len(counts))]
        axes[0].plot(ts,counts,color=A1,lw=1.5,alpha=0.9)
        axes[0].fill_between(ts,counts,alpha=0.10,color=A1)
        if len(counts)>10:
            ww = max(3,len(counts)//20)
            roll = np.convolve(counts,np.ones(ww)/ww,"valid")
            rt   = ts[ww//2:ww//2+len(roll)]
            axes[0].plot(rt,roll,color=A2,lw=2,ls="--",alpha=0.9,label="Smoothed average")
            axes[0].legend(facecolor=PNL,edgecolor=GRD,labelcolor=TXT,fontsize=7)
        axes[0].set(xlabel="Time (seconds)",ylabel="Number of worms",title="Worm Count Over Time")

        # Velocity
        vels=[]
        for d in all_tracks.values():
            pos=d["pos"]
            if len(pos)<2: continue
            dist=sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
            dur_t=(d["last"]-d["first"])/max(fps,1)
            if dur_t>0: vels.append(dist/dur_t)
        if vels:
            axes[1].hist(vels,bins=15,color=A1,edgecolor=BG,alpha=0.85)
            axes[1].axvline(np.mean(vels),color=A2,lw=2,ls="--",label=f"Average: {np.mean(vels):.0f}")
            axes[1].legend(facecolor=PNL,edgecolor=GRD,labelcolor=TXT,fontsize=7)
        axes[1].set(xlabel="Speed (pixels/second)",ylabel="Number of worms",title="Speed Distribution")

        # Paths
        clrs=plt.cm.tab10(np.linspace(0,1,max(1,len(all_tracks))))
        for idx,(tid,d) in enumerate(all_tracks.items()):
            pos=d["pos"]
            if len(pos)<2: continue
            xs=[p[0] for p in pos]; ys=[p[1] for p in pos]
            axes[2].plot(xs,ys,color=clrs[idx%len(clrs)],lw=0.8,alpha=0.7)
            axes[2].scatter(xs[0],ys[0],color=A2,s=12,zorder=5)
            axes[2].scatter(xs[-1],ys[-1],color="#dc2626",s=12,marker="x",lw=1.5,zorder=5)
        axes[2].set_xlim(0,W); axes[2].set_ylim(H,0)
        axes[2].set(xlabel="Horizontal position",ylabel="Vertical position",title="Movement Paths  ● start  × end")
        axes[2].set_aspect("equal","datalim")

        # Heatmap
        sc=8
        hm=np.zeros((H//sc+1,W//sc+1),dtype=np.float32)
        for d in all_tracks.values():
            for x,y,_ in d["pos"]:
                hm[min(int(y/sc),hm.shape[0]-1), min(int(x/sc),hm.shape[1]-1)] += 1
        hm = gaussian_filter(hm,sigma=3)
        im = axes[3].imshow(hm,cmap="YlOrRd",aspect="auto")
        plt.colorbar(im,ax=axes[3],shrink=0.85).ax.tick_params(labelcolor=TXT,labelsize=7)
        axes[3].set(title="Activity Heatmap — where worms spent most time",xlabel="X",ylabel="Y")

        fig.suptitle("Movement Analysis",fontsize=12,color=TXT,fontweight="bold",y=0.97)
        out=os.path.join(tempfile.gettempdir(),"analytics.png")
        plt.savefig(out,dpi=120,facecolor=BG,bbox_inches="tight"); plt.close()
        img=cv2.imread(out)
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Chart error: {e}"); return None

# ─────────────────────────────────────────────────────────
#  CENTROID TRACKER
# ─────────────────────────────────────────────────────────
class Tracker:
    def __init__(self,max_gone=15,max_dist=80):
        self.nid=1; self.objs={}; self.bxs={}; self.gone={}
        self.mg=max_gone; self.md=max_dist

    def update(self,dets):
        if len(dets)==0:
            for oid in list(self.gone):
                self.gone[oid]+=1
                if self.gone[oid]>self.mg:
                    del self.objs[oid],self.bxs[oid],self.gone[oid]
            return []
        cens=np.array([((d[0]+d[2])/2,(d[1]+d[3])/2) for d in dets])
        if not self.objs:
            res=[]
            for d,c in zip(dets,cens):
                self.objs[self.nid]=c; self.bxs[self.nid]=d; self.gone[self.nid]=0
                res.append((self.nid,d)); self.nid+=1
            return res
        from scipy.spatial.distance import cdist
        ids=list(self.objs); oc=np.array([self.objs[i] for i in ids])
        D=cdist(oc,cens)
        rows=D.min(1).argsort(); cols=D.argmin(1)[rows]
        ur,uc=set(),set(); res=[]
        for r,c in zip(rows,cols):
            if r in ur or c in uc or D[r,c]>self.md: continue
            oid=ids[r]; self.objs[oid]=cens[c]; self.bxs[oid]=dets[c]; self.gone[oid]=0
            ur.add(r); uc.add(c); res.append((oid,dets[c]))
        for c in range(len(cens)):
            if c not in uc:
                self.objs[self.nid]=cens[c]; self.bxs[self.nid]=dets[c]; self.gone[self.nid]=0
                res.append((self.nid,dets[c])); self.nid+=1
        for r in range(len(ids)):
            if r not in ur:
                oid=ids[r]; self.gone[oid]+=1
                if self.gone[oid]>self.mg: del self.objs[oid],self.bxs[oid],self.gone[oid]
        return res

# ─────────────────────────────────────────────────────────
#  CANCEL FLAG  (shared mutable object for stop button)
# ─────────────────────────────────────────────────────────
class CancelFlag:
    def __init__(self): self._stop=False
    def stop(self): self._stop=True
    def reset(self): self._stop=False
    def cancelled(self): return self._stop

CANCEL = CancelFlag()

# ─────────────────────────────────────────────────────────
#  TAB 1 — SINGLE IMAGE
# ─────────────────────────────────────────────────────────
def analyze_image(image, conf, speed):
    if image is None: return None,"Upload a photo above, then click Scan.",None
    overlap = SPEED_OPTS[speed]["overlap"]
    t0=time.time()
    boxes,scores = run_detection(image,conf,IOU_DEF,overlap)
    elapsed=time.time()-t0
    n=len(boxes)
    H,W=image.shape[:2]
    xs,ys=_steps(W,PATCH,overlap),_steps(H,PATCH,overlap)
    patches=(len(xs)*len(ys)) if (W>PATCH*1.5 or H>PATCH*1.5) else 1

    out = annotate(image,boxes,scores)
    out = stamp(out, f"{n} worms found",
                f"{elapsed:.1f}s · {patches} scan zones · {speed.split('—')[0].strip()} · {HW['dev']}")

    if n==0:
        return out,"No worms detected. Try lowering the sensitivity slider.",None

    avg_c=np.mean(scores); ww=boxes[:,2]-boxes[:,0]; hh=boxes[:,3]-boxes[:,1]
    info = (f"**Found {n} worms** in {elapsed:.1f}s\n\n"
            f"Average confidence: {avg_c:.0%}  ·  "
            f"Average size: {np.mean(ww):.0f}×{np.mean(hh):.0f} pixels")

    csv_p=os.path.join(tempfile.gettempdir(),"detections.csv")
    with open(csv_p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["worm_number","x1","y1","x2","y2","confidence","width_px","height_px"])
        for i,(b,s) in enumerate(zip(boxes,scores)):
            x1,y1,x2,y2=b
            w.writerow([i+1,f"{x1:.0f}",f"{y1:.0f}",f"{x2:.0f}",f"{y2:.0f}",
                        f"{s:.3f}",f"{x2-x1:.0f}",f"{y2-y1:.0f}"])
    return out,info,csv_p

# ─────────────────────────────────────────────────────────
#  TAB 2 — VIDEO TRACKING  (with stop support)
# ─────────────────────────────────────────────────────────
def start_tracking(video_path, conf, speed, trail_len,
                   cx1,cy1,cx2,cy2, trim_s,trim_e,
                   progress=gr.Progress()):
    CANCEL.reset()
    if not video_path: return None,"Upload a video first.",None,None

    cap=cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total/fps

    # Crop
    cx1,cy1,cx2,cy2=int(cx1),int(cy1),int(cx2),int(cy2)
    use_crop=not(cx1==0 and cy1==0 and cx2==0 and cy2==0)
    if use_crop:
        cx1=max(0,min(cx1,ow-1)); cy1=max(0,min(cy1,oh-1))
        cx2=max(cx1+10,min(cx2,ow)); cy2=max(cy1+10,min(cy2,oh))
    else:
        cx1,cy1,cx2,cy2=0,0,ow,oh
    W,H=cx2-cx1, cy2-cy1

    # Trim
    ts=max(0.0,float(trim_s)); te=float(trim_e)
    if te<=ts: te=dur
    sf=int(ts*fps); ef=min(int(te*fps),total)
    total_f=ef-sf

    preset=SPEED_OPTS[speed]
    overlap=preset["overlap"]; skip=max(1,preset["skip"])
    to_proc=max(1,total_f//skip)

    out_p=os.path.join(tempfile.gettempdir(),"tracked.mp4")
    writer=cv2.VideoWriter(out_p,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))

    tracker=Tracker(max_gone=int(fps*2))
    counts=[]; all_tracks={}; trails=defaultdict(list)
    last_ann=None; processed=0; t0=time.time(); eta_val=999

    if sf>0: cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    progress(0,desc="Starting…")

    for fn in range(total_f):
        if CANCEL.cancelled(): break
        ok,frame=cap.read()
        if not ok: break
        if use_crop: frame=frame[cy1:cy2,cx1:cx2]

        if fn%skip==0:
            boxes,scores=run_detection(frame,conf,IOU_DEF,overlap)
            tracked=tracker.update(boxes)
            tids,tboxes=[],[]
            for tid,box in tracked:
                tids.append(tid); tboxes.append(box)
                cx_p=(box[0]+box[2])/2; cy_p=(box[1]+box[3])/2
                if tid not in all_tracks: all_tracks[tid]={"first":fn,"pos":[]}
                all_tracks[tid]["last"]=fn
                all_tracks[tid]["pos"].append((cx_p,cy_p,fn))
                trails[tid].append((cx_p,cy_p))
                if len(trails[tid])>trail_len: trails[tid]=trails[tid][-trail_len:]
            tboxes  = np.array(tboxes)  if tboxes  else np.array([]).reshape(0,4)
            tscores = scores[:len(tboxes)] if len(scores)>=len(tboxes) else np.ones(len(tboxes))
            tids_a  = np.array(tids) if tids else None
            counts.append(len(tboxes))
            ann=annotate(frame,tboxes,tscores,tids_a,trails)
            ann=stamp(ann,
                      f"{len(tboxes)} worms  ·  {len(all_tracks)} unique",
                      f"t={fn/fps:.1f}s  ·  ETA {fmt_time(eta_val)}")
            last_ann=ann; processed+=1
            if processed>1:
                eta_val=(to_proc-processed)*(time.time()-t0)/processed
        else:
            ann=last_ann if last_ann is not None else frame

        writer.write(ann)
        if fn%20==0:
            progress(fn/total_f, desc=f"Frame {fn}/{total_f}  ·  ETA {fmt_time(eta_val)}")

    cap.release(); writer.release()

    if CANCEL.cancelled():
        return out_p,"⏹ Tracking stopped early — partial results shown below.",None,None

    # CSV
    csv_p=os.path.join(tempfile.gettempdir(),"tracking.csv")
    with open(csv_p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["worm_id","first_frame","last_frame","duration_s",
                    "total_distance_px","avg_speed_px_per_s","positions_recorded"])
        for tid,d in sorted(all_tracks.items()):
            pos=d["pos"]; first=d["first"]; last_f=d["last"]
            dur_t=(last_f-first)/fps
            dist=sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
            w.writerow([tid,first,last_f,f"{dur_t:.2f}",f"{dist:.0f}",
                        f"{dist/dur_t:.1f}" if dur_t>0 else "0",len(pos)])

    total_t=time.time()-t0
    vels,dists=[],[]
    for d in all_tracks.values():
        pos=d["pos"]
        if len(pos)<2: continue
        dist=sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
        dur_t=(d["last"]-d["first"])/fps
        dists.append(dist)
        if dur_t>0: vels.append(dist/dur_t)

    avg_vel = f"{np.mean(vels):.0f} px/s" if vels else "—"
    avg_dist= f"{np.mean(dists):.0f} px"  if dists else "—"

    info=(f"**Tracking complete** · {len(all_tracks)} unique worms · finished in {fmt_time(total_t)}\n\n"
          f"| | |\n|---|---|\n"
          f"| Video length | {total_f/fps:.1f}s at {fps:.0f} fps |\n"
          f"| Average worms per frame | {np.mean(counts):.1f} |\n"
          f"| Most worms at once | {max(counts) if counts else 0} |\n"
          f"| Average speed | {avg_vel} |\n"
          f"| Average distance moved | {avg_dist} |")

    chart=make_chart(all_tracks,counts,fps,W,H)
    return out_p,info,csv_p,chart

def stop_tracking():
    CANCEL.stop()
    return "⏹ Stop signal sent — finishing current frame…"

# ─────────────────────────────────────────────────────────
#  TAB 3 — BATCH
# ─────────────────────────────────────────────────────────
def run_batch(files,conf,speed,progress=gr.Progress()):
    if not files: return None,"Upload some images first.",None
    overlap=SPEED_OPTS[speed]["overlap"]
    out_dir=os.path.join(tempfile.gettempdir(),"worm_batch")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rows=[]; gallery=[]
    for i,fp in enumerate(files):
        progress((i+1)/len(files),desc=f"Image {i+1} of {len(files)}")
        img=cv2.imread(fp)
        if img is None: continue
        fname=os.path.basename(fp)
        boxes,scores=run_detection(img,conf,IOU_DEF,overlap)
        ann=annotate(img,boxes,scores)
        cv2.imwrite(os.path.join(out_dir,f"detected_{fname}"),ann)
        gallery.append((cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),f"{fname}  —  {len(boxes)} worms"))
        rows.append({"file":fname,"count":len(boxes),
                     "avg_conf":f"{np.mean(scores):.0%}" if len(scores) else "—"})

    csv_p=os.path.join(out_dir,"summary.csv")
    with open(csv_p,"w",newline="") as f:
        dw=csv.DictWriter(f,["file","count","avg_conf"]); dw.writeheader(); dw.writerows(rows)

    zip_p=shutil.make_archive(os.path.join(tempfile.gettempdir(),"worm_batch"),"zip",out_dir)
    total=sum(r["count"] for r in rows)
    avg  =total/len(rows) if rows else 0

    info=(f"**{len(rows)} images processed**  ·  {total} worms found  ·  {avg:.1f} average per image\n\n"
          "| File | Worms | Confidence |\n|---|---|---|\n"+
          "\n".join(f"| {r['file']} | {r['count']} | {r['avg_conf']} |" for r in rows))
    return gallery,info,zip_p

# ─────────────────────────────────────────────────────────
#  TAB 4 — FRAME EXPLORER
# ─────────────────────────────────────────────────────────
def analyze_frame(video_path,conf,frame_no,cx1,cy1,cx2,cy2):
    if not video_path: return None,"Upload a video first."
    cap=cv2.VideoCapture(video_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps  =cap.get(cv2.CAP_PROP_FPS) or 30
    ow   =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh   =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fn   =min(max(0,int(frame_no)),total-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES,fn); ok,frame=cap.read(); cap.release()
    if not ok: return None,"Cannot read that frame."

    cx1,cy1,cx2,cy2=int(cx1),int(cy1),int(cx2),int(cy2)
    if not(cx1==0 and cy1==0 and cx2==0 and cy2==0):
        cx1=max(0,min(cx1,ow-1)); cy1=max(0,min(cy1,oh-1))
        cx2=max(cx1+10,min(cx2,ow)); cy2=max(cy1+10,min(cy2,oh))
        frame=frame[cy1:cy2,cx1:cx2]

    boxes,scores=run_detection(frame,conf)
    ann=annotate(frame,boxes,scores)
    ann=stamp(ann,f"{len(boxes)} worms",f"frame {fn}  ·  {fn/fps:.2f}s into video")
    info=(f"**Frame {fn}** ({fn/fps:.2f}s)  —  **{len(boxes)} worms**"
          +(f"  ·  avg confidence {np.mean(scores):.0%}" if len(scores) else ""))
    return cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),info

# ─────────────────────────────────────────────────────────
#  TAB 5 — TRIM & CROP TOOL
# ─────────────────────────────────────────────────────────
def apply_trim_crop(path,x1,y1,x2,y2,ts,te,progress=gr.Progress()):
    if not path: return None,"Upload a video first.",None
    cap=cv2.VideoCapture(path)
    fps  =cap.get(cv2.CAP_PROP_FPS) or 30
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow   =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh   =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur  =total/fps

    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    x1=max(0,min(x1,ow-1)); y1=max(0,min(y1,oh-1))
    x2=max(x1+10,min(x2,ow)); y2=max(y1+10,min(y2,oh))
    ts=max(0.0,float(ts)); te=float(te)
    if te<=ts: te=dur
    sf=int(ts*fps); ef=int(te*fps); nf=ef-sf
    CW,CH=x2-x1,y2-y1

    out_p=os.path.join(tempfile.gettempdir(),"trimmed_cropped.mp4")
    wt=cv2.VideoWriter(out_p,cv2.VideoWriter_fourcc(*"mp4v"),fps,(CW,CH))
    cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    t0=time.time(); done=0
    for _ in range(nf):
        ok,frame=cap.read()
        if not ok: break
        wt.write(frame[y1:y2,x1:x2]); done+=1
        if done%100==0:
            el=time.time()-t0; rate=done/el if el>0 else 1
            progress(done/nf,desc=f"Frame {done}/{nf}  ·  ETA {fmt_time((nf-done)/rate)}")
    cap.release(); wt.release()

    tt=time.time()-t0
    speedup=(ow*oh)/(CW*CH) if CW*CH else 1
    info=(f"**Done!**  Output: {CW}×{CH}px  ·  {te-ts:.1f}s  ·  "
          f"processed in {tt:.1f}s\n\n"
          f"This video is **{speedup:.1f}× smaller** — tracking it will be that much faster.")
    return out_p,info,out_p

# ═══════════════════════════════════════════════════════════
#  CSS  — fluid, responsive, no unnecessary cards
# ═══════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

:root {
    --bg:        #f0f2f5;
    --surface:   #ffffff;
    --border:    #dde1e7;
    --text:      #141619;
    --muted:     #5c6370;
    --faint:     #8b919e;
    --accent:    #1d6ef5;
    --accent-dk: #1558d6;
    --green:     #0f9b5c;
    --red:       #d94040;
    --sidebar-w: clamp(260px, 26vw, 360px);
    --gap:       clamp(12px, 2vw, 24px);
    --r:         10px;
    --font:      'DM Sans', system-ui, sans-serif;
}

/* ── Reset & base ─────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: clamp(13px, 1.1vw, 15px) !important;
}

/* ── App bar ──────────────────────────── */
.appbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 clamp(16px, 3vw, 32px);
    height: 52px;
    position: sticky; top: 0; z-index: 200;
}
.appbar-logo {
    width: 30px; height: 30px; flex-shrink: 0;
    background: linear-gradient(135deg,#1d6ef5,#06b6d4);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
}
.appbar h1 {
    font-size: clamp(14px,1.3vw,16px) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin: 0 !important; letter-spacing: -0.2px;
}
.appbar-chip {
    font-size: 10px; font-weight: 600;
    color: var(--accent); background: #eff4ff;
    border: 1px solid #c3d5fc; padding: 2px 7px; border-radius: 99px;
}
.appbar-hw {
    margin-left: auto;
    font-size: clamp(10px,0.9vw,12px);
    color: var(--faint);
    white-space: nowrap;
}

/* ── Tabs ─────────────────────────────── */
.tab-nav {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 clamp(12px,2.5vw,28px) !important;
    gap: 0 !important; margin: 0 !important;
}
.tab-nav button {
    background: transparent !important; border: none !important;
    border-bottom: 2px solid transparent !important;
    color: var(--muted) !important;
    font-family: var(--font) !important;
    font-size: clamp(12px,1vw,14px) !important;
    font-weight: 500 !important;
    padding: 13px clamp(10px,1.5vw,18px) !important;
    border-radius: 0 !important; cursor: pointer !important;
    transition: color .15s !important;
}
.tab-nav button:hover { color: var(--text) !important; }
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Tab content area ─────────────────── */
.gradio-tabitem, div[class*="tabitem"] {
    padding: clamp(14px,2vw,24px) clamp(14px,2.5vw,28px) !important;
    background: var(--bg) !important;
}

/* ── NO CARD borders on blocks by default ── */
.block, .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

/* ── Only the main row panels get a surface ── */
.main-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: clamp(14px,1.8vw,22px) !important;
}

/* ── Sidebar ──────────────────────────── */
.sidebar {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: clamp(14px,1.8vw,20px) !important;
}

/* ── Section labels inside sidebar ───── */
.sec-label {
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .9px !important;
    color: var(--faint) !important;
    padding-top: 14px !important;
    margin-top: 6px !important;
    border-top: 1px solid var(--border) !important;
    display: block;
}
.sec-label:first-child { padding-top: 0 !important; border-top: none !important; }

/* ── Labels ───────────────────────────── */
label > span, .label-wrap > span {
    font-size: clamp(11px,0.9vw,13px) !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    font-family: var(--font) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Inputs ───────────────────────────── */
input[type=number], input[type=text], textarea, select {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-size: clamp(12px,0.95vw,14px) !important;
    font-family: var(--font) !important;
    transition: border-color .15s, box-shadow .15s !important;
}
input[type=number]:focus, input[type=text]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(29,110,245,.10) !important;
    outline: none !important;
}
input[type=range] { accent-color: var(--accent) !important; }
input[type=checkbox], input[type=radio] { accent-color: var(--accent) !important; }

/* ── Buttons ──────────────────────────── */
button.primary {
    background: var(--accent) !important;
    color: #fff !important; border: none !important;
    border-radius: 7px !important;
    font-family: var(--font) !important;
    font-size: clamp(12px,1vw,14px) !important;
    font-weight: 600 !important;
    padding: clamp(8px,0.8vw,11px) clamp(14px,1.5vw,20px) !important;
    cursor: pointer !important;
    transition: background .15s, transform .1s, box-shadow .15s !important;
}
button.primary:hover {
    background: var(--accent-dk) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(29,110,245,.25) !important;
}
button.primary:active { transform: translateY(0) !important; }

/* Stop button */
button.stop-btn {
    background: var(--red) !important;
    color: #fff !important; border: none !important;
    border-radius: 7px !important;
    font-family: var(--font) !important;
    font-size: clamp(12px,1vw,14px) !important;
    font-weight: 600 !important;
    padding: clamp(8px,0.8vw,11px) clamp(14px,1.5vw,20px) !important;
    cursor: pointer !important;
    transition: background .15s !important;
}
button.stop-btn:hover { background: #b83030 !important; }

button.secondary {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    font-family: var(--font) !important;
    font-size: clamp(12px,1vw,14px) !important;
    font-weight: 500 !important;
    padding: clamp(7px,0.7vw,10px) clamp(12px,1.2vw,16px) !important;
    cursor: pointer !important;
    transition: border-color .15s, color .15s !important;
}
button.secondary:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* ── Upload zones ─────────────────────── */
.upload-button, .upload-zone,
div[data-testid="image"] > div:first-child,
div[data-testid="video"] > div:first-child {
    background: var(--bg) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    transition: border-color .2s, background .2s !important;
}
.upload-zone:hover,
div[data-testid="image"] > div:first-child:hover,
div[data-testid="video"] > div:first-child:hover {
    border-color: var(--accent) !important;
    background: #f5f8ff !important;
}

/* ── Accordion ────────────────────────── */
.accordion {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    overflow: hidden !important;
}
.accordion > .label-wrap {
    padding: 10px 14px !important;
    font-size: clamp(12px,0.95vw,13px) !important;
    font-weight: 500 !important;
}

/* ── Markdown ─────────────────────────── */
.prose, .md {
    font-size: clamp(12px,0.95vw,14px) !important;
    line-height: 1.65 !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
.prose p, .md p { margin: 0 0 6px !important; }
.prose h2, .md h2 {
    font-size: clamp(13px,1.05vw,15px) !important;
    font-weight: 600 !important; margin: 0 0 8px !important;
}
.prose strong, .md strong { color: var(--green) !important; font-weight: 600 !important; }
.prose table, .md table {
    width: 100% !important; border-collapse: collapse !important;
    font-size: clamp(11px,0.9vw,13px) !important; margin-top: 8px !important;
}
.prose th, .md th {
    background: var(--bg) !important; padding: 5px 9px !important;
    font-size: 10px !important; font-weight: 600 !important;
    color: var(--faint) !important; text-align: left !important;
    text-transform: uppercase !important; letter-spacing: .4px !important;
}
.prose td, .md td { padding: 5px 9px !important; border-bottom: 1px solid var(--border) !important; }
.prose tr:last-child td, .md tr:last-child td { border: none !important; }

/* ── Gallery ──────────────────────────── */
.gallery-item { border-radius: 6px !important; overflow: hidden !important; }

/* ── Scrollbars ───────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Footer ───────────────────────────── */
footer { display: none !important; }
.app-footer {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 10px clamp(14px,2.5vw,28px);
    font-size: clamp(10px,0.85vw,12px);
    color: var(--faint);
    display: flex; justify-content: space-between; align-items: center;
}

/* ── Status bar (stop message) ────────── */
.status-bar {
    font-size: clamp(11px,0.9vw,13px) !important;
    color: var(--muted) !important;
    padding: 6px 0 !important;
    min-height: 24px;
}
"""

# ═══════════════════════════════════════════════════════════
#  BUILD UI
# ═══════════════════════════════════════════════════════════
with gr.Blocks(title="Worm Analyzer", css=CSS) as app:

    # ── App bar
    gr.HTML(f"""
    <div class="appbar">
      <div class="appbar-logo">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round">
          <path d="M3 12c0-3 2-5 4-5s3 2 5 2 3-2 5-2 4 2 4 5-2 5-4 5-3-2-5-2-3 2-5 2-4-2-4-5z"/>
        </svg>
      </div>
      <h1>Worm Analyzer</h1>
      <span class="appbar-chip">YOLOv8m</span>
      <span class="appbar-hw">{HW['dev']} &nbsp;·&nbsp; {HW['name'][:28]} &nbsp;·&nbsp; {HW['ms']:.0f}ms per scan</span>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════
        # TAB 1  ·  Scan a photo
        # ══════════════════════════════════════════
        with gr.Tab("📷  Scan Photo"):
            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(elem_classes=["sidebar"], scale=3, min_width=0):
                    gr.HTML('<span class="sec-label">Photo</span>')
                    img_in = gr.Image(type="numpy", label="Drop a photo here or click to upload",
                                      height=260, show_label=False)

                    gr.HTML('<span class="sec-label">Settings</span>')
                    img_speed = gr.Radio(
                        list(SPEED_OPTS.keys()), value=DEF_SPEED,
                        label="Scan quality",
                        info="Higher quality finds more worms but takes longer"
                    )
                    img_conf = gr.Slider(0.1, 0.9, CONF_DEF, step=0.05,
                                         label="Sensitivity  (lower = finds more, higher = fewer false detections)")
                    img_btn  = gr.Button("🔍  Scan for Worms", variant="primary", size="lg")

                # Results
                with gr.Column(elem_classes=["main-panel"], scale=7, min_width=0):
                    img_out  = gr.Image(label="Result", height=400, show_label=False)
                    img_info = gr.Markdown("*Upload a photo on the left and click **Scan for Worms**.*")
                    img_csv  = gr.File(label="Download results as spreadsheet", visible=True,
                                       elem_classes=["block"])

            img_btn.click(analyze_image, [img_in, img_conf, img_speed],
                          [img_out, img_info, img_csv])

        # ══════════════════════════════════════════
        # TAB 2  ·  Track in video
        # ══════════════════════════════════════════
        with gr.Tab("🎬  Track in Video"):
            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(elem_classes=["sidebar"], scale=3, min_width=0):
                    gr.HTML('<span class="sec-label">Video</span>')
                    vt_vid  = gr.Video(label="Upload your video", height=200, show_label=False)
                    vt_info = gr.Markdown("*Upload a video to get started.*",
                                          elem_classes=["status-bar"])

                    gr.HTML('<span class="sec-label">Scan area  —  focus on the dish</span>')
                    with gr.Accordion("📐  Crop settings", open=True):
                        with gr.Row():
                            vt_auto = gr.Checkbox(label="Auto-detect dish", value=True)
                            vt_pad  = gr.Slider(-50, 150, 50, step=5,
                                                label="Border padding (pixels)")
                        vt_prev = gr.Image(label=None, height=150, show_label=False)
                        vt_x1 = gr.Number(0, visible=False, precision=0)
                        vt_y1 = gr.Number(0, visible=False, precision=0)
                        vt_x2 = gr.Number(0, visible=False, precision=0)
                        vt_y2 = gr.Number(0, visible=False, precision=0)

                    gr.HTML('<span class="sec-label">Trim  —  only process part of the video</span>')
                    with gr.Row():
                        vt_ts = gr.Number(0,   label="Start at (seconds)", precision=1)
                        vt_te = gr.Number(0,   label="Stop at (seconds, 0 = end)", precision=1)

                    gr.HTML('<span class="sec-label">Quality & detail</span>')
                    vt_speed = gr.Radio(list(SPEED_OPTS.keys()), value=DEF_SPEED,
                                        label="Processing speed")
                    vt_conf  = gr.Slider(0.1,0.9,CONF_DEF,step=0.05,label="Sensitivity")
                    vt_trail = gr.Slider(10,200,60,step=10,
                                         label="Trail length  (how long the movement path stays visible)")

                    with gr.Row():
                        vt_btn  = gr.Button("▶  Start Tracking", variant="primary")
                        vt_stop = gr.Button("⏹  Stop", elem_classes=["stop-btn"])
                    vt_stop_msg = gr.Markdown("", elem_classes=["status-bar"])

                # Results
                with gr.Column(elem_classes=["main-panel"], scale=7, min_width=0):
                    vt_out  = gr.Video(label="Tracked video", height=360, show_label=False)
                    vt_stat = gr.Markdown("*Results will appear here after tracking.*")
                    vt_chart= gr.Image(label=None, height=260, show_label=False)
                    vt_csv  = gr.File(label="Download tracking data as spreadsheet",
                                      elem_classes=["block"])

            vt_dd  = gr.State()
            vt_frm = gr.State()

            def _vt_upload(p):
                dd,rgb,x1,y1,x2,y2,info,dur = load_video(p)
                prev = make_preview(dd,rgb,x1,y1,x2,y2) if dd else None
                return dd,rgb,x1,y1,x2,y2,info,prev,0,dur

            def _vt_crop(dd,rgb,auto,pad):
                x1,y1,x2,y2,prev = update_crop(dd,rgb,auto,pad)
                return x1,y1,x2,y2,prev

            vt_vid.change(_vt_upload,[vt_vid],
                          [vt_dd,vt_frm,vt_x1,vt_y1,vt_x2,vt_y2,vt_info,vt_prev,vt_ts,vt_te])
            vt_auto.change(_vt_crop,[vt_dd,vt_frm,vt_auto,vt_pad],[vt_x1,vt_y1,vt_x2,vt_y2,vt_prev])
            vt_pad.change( _vt_crop,[vt_dd,vt_frm,vt_auto,vt_pad],[vt_x1,vt_y1,vt_x2,vt_y2,vt_prev])
            vt_btn.click(start_tracking,
                         [vt_vid,vt_conf,vt_speed,vt_trail,
                          vt_x1,vt_y1,vt_x2,vt_y2,vt_ts,vt_te],
                         [vt_out,vt_stat,vt_csv,vt_chart])
            vt_stop.click(stop_tracking, [], [vt_stop_msg])

        # ══════════════════════════════════════════
        # TAB 3  ·  Batch — many photos at once
        # ══════════════════════════════════════════
        with gr.Tab("📁  Batch — Many Photos"):
            with gr.Row(equal_height=False):

                with gr.Column(elem_classes=["sidebar"], scale=3, min_width=0):
                    gr.HTML('<span class="sec-label">Photos</span>')
                    bt_files = gr.File(file_count="multiple", file_types=["image"],
                                       label="Select all photos at once", show_label=False)
                    gr.HTML('<span class="sec-label">Settings</span>')
                    bt_speed = gr.Radio(list(SPEED_OPTS.keys()), value=DEF_SPEED,
                                        label="Scan quality")
                    bt_conf  = gr.Slider(0.1,0.9,CONF_DEF,step=0.05,label="Sensitivity")
                    bt_btn   = gr.Button("🔍  Scan All Photos", variant="primary", size="lg")
                    bt_info  = gr.Markdown("*Upload photos and click Scan All.*",
                                           elem_classes=["status-bar"])
                    bt_zip   = gr.File(label="Download all results (ZIP)", elem_classes=["block"])

                with gr.Column(elem_classes=["main-panel"], scale=7, min_width=0):
                    bt_gallery = gr.Gallery(label=None, columns=4, height=500, show_label=False)

            bt_btn.click(run_batch,[bt_files,bt_conf,bt_speed],[bt_gallery,bt_info,bt_zip])

        # ══════════════════════════════════════════
        # TAB 4  ·  Explore frame by frame
        # ══════════════════════════════════════════
        with gr.Tab("🔬  Frame Explorer"):
            with gr.Row(equal_height=False):

                with gr.Column(elem_classes=["sidebar"], scale=3, min_width=0):
                    gr.HTML('<span class="sec-label">Video</span>')
                    fe_vid   = gr.Video(label="Upload a video", height=200, show_label=False)
                    gr.HTML('<span class="sec-label">Browse frames</span>')
                    fe_frame = gr.Slider(0,1000,0,step=1,
                                         label="Frame position  (drag to move through the video)")
                    fe_conf  = gr.Slider(0.1,0.9,CONF_DEF,step=0.05,label="Sensitivity")
                    with gr.Accordion("📐  Crop settings", open=False):
                        fe_msg = gr.Markdown("*Upload a video to enable auto-crop.*")
                        with gr.Row():
                            fe_auto = gr.Checkbox(label="Auto-crop dish", value=True)
                            fe_pad  = gr.Slider(-50,150,50,step=5,label="Padding")
                        fe_x1 = gr.Number(0,visible=False,precision=0)
                        fe_y1 = gr.Number(0,visible=False,precision=0)
                        fe_x2 = gr.Number(0,visible=False,precision=0)
                        fe_y2 = gr.Number(0,visible=False,precision=0)
                    fe_btn   = gr.Button("🔍  Analyze This Frame", variant="primary", size="lg")

                with gr.Column(elem_classes=["main-panel"], scale=7, min_width=0):
                    fe_out  = gr.Image(label=None, height=460, show_label=False)
                    fe_info = gr.Markdown("*Select a frame and click Analyze.*")

            fe_dd  = gr.State()
            fe_fst = gr.State()

            def _fe_upload(p):
                dd,rgb,x1,y1,x2,y2,info,_ = load_video(p)
                cap=cv2.VideoCapture(p) if p else None
                total=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 1)
                if cap: cap.release()
                return gr.Slider(maximum=max(1,total-1)),dd,rgb,info,x1,y1,x2,y2

            def _fe_crop(dd,rgb,auto,pad):
                x1,y1,x2,y2,_=update_crop(dd,rgb,auto,pad); return x1,y1,x2,y2

            fe_vid.change(_fe_upload,[fe_vid],[fe_frame,fe_dd,fe_fst,fe_msg,fe_x1,fe_y1,fe_x2,fe_y2])
            fe_auto.change(_fe_crop,[fe_dd,fe_fst,fe_auto,fe_pad],[fe_x1,fe_y1,fe_x2,fe_y2])
            fe_pad.change( _fe_crop,[fe_dd,fe_fst,fe_auto,fe_pad],[fe_x1,fe_y1,fe_x2,fe_y2])
            fe_btn.click(analyze_frame,
                         [fe_vid,fe_conf,fe_frame,fe_x1,fe_y1,fe_x2,fe_y2],
                         [fe_out,fe_info])

        # ══════════════════════════════════════════
        # TAB 5  ·  Prepare video (trim & crop)
        # ══════════════════════════════════════════
        with gr.Tab("✂️  Prepare Video"):
            gr.Markdown(
                "**Trim and crop your video before tracking.** "
                "A smaller, shorter video scans much faster — "
                "use this if you only need part of a recording.",
                elem_classes=["prose"]
            )
            with gr.Row(equal_height=False):

                with gr.Column(elem_classes=["sidebar"], scale=3, min_width=0):
                    gr.HTML('<span class="sec-label">Video</span>')
                    vl_vid  = gr.Video(label="Upload video", height=200, show_label=False)
                    vl_info = gr.Markdown("*Upload a video to begin.*", elem_classes=["status-bar"])

                    gr.HTML('<span class="sec-label">Crop  —  remove area outside dish</span>')
                    with gr.Row():
                        vl_auto = gr.Checkbox(label="Auto-detect dish", value=True)
                        vl_pad  = gr.Slider(-50,150,50,step=5,label="Padding")
                    vl_x1 = gr.Number(0,visible=False,precision=0)
                    vl_y1 = gr.Number(0,visible=False,precision=0)
                    vl_x2 = gr.Number(1920,visible=False,precision=0)
                    vl_y2 = gr.Number(1080,visible=False,precision=0)

                    gr.HTML('<span class="sec-label">Trim  —  cut to the section you need</span>')
                    with gr.Row():
                        vl_ts = gr.Number(0, label="Start at (seconds)", precision=1)
                        vl_te = gr.Number(0, label="End at (seconds, 0 = end)", precision=1)

                    vl_btn = gr.Button("✂️  Apply Crop & Trim", variant="primary", size="lg")

                with gr.Column(elem_classes=["main-panel"], scale=7, min_width=0):
                    with gr.Tabs():
                        with gr.Tab("Preview — shows what will be kept"):
                            vl_prev = gr.Image(label=None, height=360, show_label=False)
                        with gr.Tab("Result — processed video"):
                            vl_out  = gr.Video(label=None, height=360, show_label=False)
                    vl_stat = gr.Markdown("*Crop and trim settings, then click Apply.*")
                    vl_dl   = gr.File(label="Download processed video", elem_classes=["block"])

            vl_dd  = gr.State()
            vl_frm = gr.State()

            def _vl_upload(p):
                dd,rgb,x1,y1,x2,y2,info,dur = load_video(p)
                prev=make_preview(dd,rgb,x1,y1,x2,y2) if dd else None
                return dd,rgb,info,x1,y1,x2,y2,0,dur,prev

            def _vl_crop(dd,rgb,auto,pad):
                x1,y1,x2,y2,prev=update_crop(dd,rgb,auto,pad); return x1,y1,x2,y2,prev

            vl_vid.change(_vl_upload,[vl_vid],
                          [vl_dd,vl_frm,vl_info,vl_x1,vl_y1,vl_x2,vl_y2,vl_ts,vl_te,vl_prev])
            vl_auto.change(_vl_crop,[vl_dd,vl_frm,vl_auto,vl_pad],[vl_x1,vl_y1,vl_x2,vl_y2,vl_prev])
            vl_pad.change( _vl_crop,[vl_dd,vl_frm,vl_auto,vl_pad],[vl_x1,vl_y1,vl_x2,vl_y2,vl_prev])
            vl_btn.click(apply_trim_crop,
                         [vl_vid,vl_x1,vl_y1,vl_x2,vl_y2,vl_ts,vl_te],
                         [vl_out,vl_stat,vl_dl])

        # ══════════════════════════════════════════
        # TAB 6  ·  Guide
        # ══════════════════════════════════════════
        with gr.Tab("❓  Guide"):
            with gr.Row():
                with gr.Column(elem_classes=["main-panel"]):
                    gr.Markdown(f"""
## How to use each tab

**📷 Scan Photo** — drop in a single photo. The app finds all worms and marks them with coloured brackets. Download the spreadsheet for exact positions.

**🎬 Track in Video** — upload a video and the app follows every worm through the whole recording, giving each one a unique number. It auto-detects the petri dish so it only scans the relevant area. Hit **Stop** at any time to get partial results.

**📁 Batch — Many Photos** — process a whole folder of images at once. Download a ZIP with every annotated image plus a summary spreadsheet.

**🔬 Frame Explorer** — step through a video one frame at a time. Useful for checking whether your sensitivity setting is right before starting a full tracking run.

**✂️ Prepare Video** — trim your video to the section you need and crop out everything outside the dish. A smaller video tracks much faster. Use this first if you have long recordings.

---
## Sensitivity setting

The sensitivity slider controls how confident the AI must be before marking something as a worm.

- **Lower (0.2–0.35)** — finds more worms including faint or small ones, but may occasionally mark non-worms
- **Middle (0.35–0.5)** — good balance for most experiments *(default)*
- **Higher (0.5–0.7)** — only very clear worms, very few false detections

---
## Scan quality

| Setting | What it does | Best for |
|---|---|---|
| Fast — preview only | Scans fewer overlapping zones, skips most frames | Quick check |
| Normal — recommended | Balanced speed and accuracy | Everyday use |
| Thorough — best results | Scans every zone, every frame | Publication data |
""")
                with gr.Column(elem_classes=["main-panel"]):
                    gr.Markdown(f"""
## Your system

| | |
|---|---|
| Computing device | {HW['dev']} ({HW['name']}) |
| Scan speed | {HW['ms']:.0f}ms per zone |
| Default quality | {DEF_SPEED} |
| AI model | YOLOv8m · 25.8M parameters |

---
## Tips for best results

**Crop to the dish** before tracking. Less background = more accurate and much faster.

**Use "Prepare Video" first** if you have a long recording — trim it to just the interesting part.

**Try Frame Explorer** to find the right sensitivity before committing to a full video run.

**Worms moving fast?** Use *Thorough* quality so the tracker doesn't lose them between frames.

**Very faint worms?** Lower sensitivity to 0.25–0.30 and use *Thorough* quality.

---
## Output files

All downloaded spreadsheets (.csv) open directly in Excel, Google Sheets, or any statistics software.

Tracking data includes: worm ID, duration seen, total distance moved, average speed, and number of times detected.
""")

    # Footer
    gr.HTML(f"""
    <div class="app-footer">
      <span>Worm Analyzer</span>
      <span>{HW['dev']} · {HW['name']} · {HW['ms']:.0f}ms/scan zone · YOLOv8m</span>
    </div>
    """)

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Worm Analyzer  →  http://localhost:7860\n")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)