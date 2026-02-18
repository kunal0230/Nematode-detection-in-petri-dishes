#!/usr/bin/env python3
"""
Worm Analyzer  ·  Detection, Tracking & Analytics
Run:  python3 worm_analyzer.py   →   http://localhost:7860
"""

import os, sys, cv2, csv, time, shutil, tempfile
import numpy as np
from collections import defaultdict
import gradio as gr
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_worm_yolov8m.pt")
PATCH      = 416
CONF_DEF   = 0.40
IOU_DEF    = 0.50
SPEED_OPTS = {
    "Fast — preview only":     {"overlap": 0.25, "skip": 6},
    "Normal — recommended":    {"overlap": 0.40, "skip": 3},
    "Thorough — best results": {"overlap": 0.50, "skip": 1},
}

# ─────────────────────────────────────────────────────────
#  MODEL + HARDWARE
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
    for _ in range(2):
        model(dummy, conf=0.5, imgsz=PATCH, verbose=False)
    t = []
    for _ in range(3):
        t0 = time.time()
        model(dummy, conf=0.5, imgsz=PATCH, verbose=False)
        t.append(time.time() - t0)
    d["ms"] = np.median(t) * 1000
    return d

print("Benchmarking hardware…")
HW = _hw()
DEF_SPEED = ("Thorough — best results" if HW["ms"] < 80  else
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
    if not s or s[-1] + patch < dim:
        s.append(max(0, dim - patch))
    return s

def run_detection(image, conf=CONF_DEF, iou=IOU_DEF, overlap=0.40):
    H, W = image.shape[:2]
    if W <= PATCH * 1.5 and H <= PATCH * 1.5:
        r = model(image, conf=conf, iou=iou, imgsz=PATCH, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
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
    bt   = torch.tensor(np.array(boxes_all),  dtype=torch.float32)
    st   = torch.tensor(np.array(scores_all), dtype=torch.float32)
    keep = torch.ops.torchvision.nms(bt, st, iou).numpy()
    return np.array(boxes_all)[keep], np.array(scores_all)[keep]

def fmt_time(s):
    if s < 60: return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"

# ─────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────
PALETTE = [(22,163,74),(37,99,235),(220,38,38),(234,88,12),
           (124,58,237),(6,182,212),(190,24,93),(5,150,105),(202,138,4),(14,116,144)]

def annotate(img, boxes, scores, ids=None, trails=None):
    out = img.copy()
    H, W = out.shape[:2]
    scale = max(0.38, min(1.1, W/1000))
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1,y1,x2,y2 = map(int, box)
        tid = int(ids[i]) if ids is not None and i < len(ids) else None
        col = PALETTE[tid % len(PALETTE)] if tid is not None else PALETTE[0]
        cl  = max(7, int((x2-x1)*0.22))
        lw  = max(1, int(2*scale))
        for px,py,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(out,(px,py),(px+dx*cl,py),col,lw+1,cv2.LINE_AA)
            cv2.line(out,(px,py),(px,py+dy*cl),col,lw+1,cv2.LINE_AA)
        roi  = out[y1:y2,x1:x2]
        fill = np.full_like(roi,col,dtype=np.uint8)
        out[y1:y2,x1:x2] = cv2.addWeighted(roi,0.90,fill,0.10,0)
        label = f"#{tid}" if tid else f"{score:.0%}"
        fs = 0.42*scale
        (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fs,1)
        ly = max(th+5,y1)
        bg = out[ly-th-4:ly,x1:x1+tw+6]
        if bg.shape[0]>0 and bg.shape[1]>0:
            out[ly-th-4:ly,x1:x1+tw+6] = cv2.addWeighted(bg,0.15,np.zeros_like(bg),0.85,0)
        cv2.putText(out,label,(x1+2,ly-3),cv2.FONT_HERSHEY_SIMPLEX,fs,col,1,cv2.LINE_AA)
    if trails:
        for tid,pts in trails.items():
            if len(pts)<2: continue
            col = PALETTE[tid % len(PALETTE)]
            for j in range(1,len(pts)):
                a = (j/len(pts))**1.5
                cv2.line(out,tuple(map(int,pts[j-1])),tuple(map(int,pts[j])),col,max(1,int(a*2)),cv2.LINE_AA)
    return out

def stamp(img, big, small):
    out = img.copy()
    fs_b,fs_s = 0.65,0.40; pad = 10
    (w1,h1),_ = cv2.getTextSize(big,cv2.FONT_HERSHEY_SIMPLEX,fs_b,2)
    (w2,h2),_ = cv2.getTextSize(small,cv2.FONT_HERSHEY_SIMPLEX,fs_s,1)
    bw = max(w1,w2)+pad*2; bh = h1+h2+pad*2+6
    roi = out[8:8+bh,8:8+bw]
    if roi.shape[0]>0 and roi.shape[1]>0:
        out[8:8+bh,8:8+bw] = cv2.addWeighted(roi,0.15,np.zeros_like(roi),0.85,0)
    cv2.putText(out,big,(8+pad,8+pad+h1),cv2.FONT_HERSHEY_SIMPLEX,fs_b,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(out,small,(8+pad,8+pad+h1+6+h2),cv2.FONT_HERSHEY_SIMPLEX,fs_s,(200,210,220),1,cv2.LINE_AA)
    return out

# ─────────────────────────────────────────────────────────
#  DISH + VIDEO HELPERS
# ─────────────────────────────────────────────────────────
def find_dish(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0,(8,8)).apply(gray)
    blur = cv2.GaussianBlur(gray,(15,15),0)
    H,W  = frame.shape[:2]
    minr,maxr = min(H,W)//6, min(H,W)//2
    for dp in [1.2,1.5,2.0]:
        for p1 in [80,50,30]:
            c = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp,min(H,W)//2,
                                 param1=p1,param2=50,minRadius=minr,maxRadius=maxr)
            if c is not None:
                best = max(np.round(c[0]).astype(int),key=lambda x:x[2])
                return int(best[0]),int(best[1]),int(best[2])
    _,thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (cx,cy),r = cv2.minEnclosingCircle(max(cnts,key=cv2.contourArea))
        if r>minr: return int(cx),int(cy),int(r)
    return None

def dish_to_crop(cx,cy,r,pad,W,H):
    if r==0: return 0,0,W,H
    p=int(pad)
    return max(0,cx-r-p),max(0,cy-r-p),min(W,cx+r+p),min(H,cy+r+p)

def load_video(path):
    if not path: return None,None,0,0,0,0,"",0
    cap  = cv2.VideoCapture(path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    total= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok,frame = cap.read(); cap.release()
    dur = total/fps
    if not ok: return None,None,0,0,W,H,"Cannot read video",dur
    rgb  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    dish = find_dish(frame)
    if dish:
        cx,cy,r = dish
        x1,y1,x2,y2 = dish_to_crop(cx,cy,r,50,W,H)
        dd   = {"cx":cx,"cy":cy,"r":r,"W":W,"H":H}
        info = f"{W}×{H}px · {dur:.1f}s · Dish detected ✓"
    else:
        x1,y1,x2,y2 = 0,0,W,H
        dd   = {"cx":0,"cy":0,"r":0,"W":W,"H":H}
        info = f"{W}×{H}px · {dur:.1f}s · No dish found"
    return dd,rgb,x1,y1,x2,y2,info,round(dur,1)

def make_preview(dd,rgb,x1,y1,x2,y2):
    if rgb is None or dd is None: return None
    prev = rgb.copy(); W,H=dd["W"],dd["H"]
    if dd["r"]>0: cv2.circle(prev,(dd["cx"],dd["cy"]),dd["r"],(37,99,235),2)
    if not(x1==0 and y1==0 and x2==W and y2==H):
        dark=(prev*0.3).astype(np.uint8); mask=np.zeros_like(prev)
        cv2.rectangle(mask,(x1,y1),(x2,y2),(255,255,255),-1)
        prev=np.where(mask>0,prev,dark)
        cv2.rectangle(prev,(x1,y1),(x2,y2),(37,99,235),3)
    return prev

def update_crop(dd,rgb,auto,pad):
    if not dd: return 0,0,0,0,None
    W,H=dd["W"],dd["H"]
    x1,y1,x2,y2 = dish_to_crop(dd["cx"],dd["cy"],dd["r"],pad,W,H) if auto else (0,0,W,H)
    return x1,y1,x2,y2,make_preview(dd,rgb,x1,y1,x2,y2)

# ─────────────────────────────────────────────────────────
#  ANALYTICS CHART
# ─────────────────────────────────────────────────────────
def make_chart(all_tracks, counts, fps, W, H):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.gridspec as mgs
        from scipy.ndimage import gaussian_filter
        BG="#fff"; PNL="#f8fafc"; TXT="#1e293b"; A1="#2563eb"; A2="#16a34a"; GRD="#e2e8f0"
        fig = plt.figure(figsize=(14,7),facecolor=BG)
        g   = mgs.GridSpec(2,3,figure=fig,hspace=0.52,wspace=0.38,left=0.07,right=0.97,top=0.91,bottom=0.10)
        axes= [fig.add_subplot(g[0,:2]),fig.add_subplot(g[0,2]),
               fig.add_subplot(g[1,:2]),fig.add_subplot(g[1,2])]
        for ax in axes:
            ax.set_facecolor(PNL)
            for sp in ax.spines.values(): sp.set_color(GRD)
            ax.tick_params(colors=TXT,labelsize=8); ax.grid(color=GRD,lw=0.6)
            ax.title.set_color(TXT); ax.title.set_fontsize(9); ax.title.set_fontweight("bold")
        ts = [i/max(fps,1) for i in range(len(counts))]
        axes[0].plot(ts,counts,color=A1,lw=1.5,alpha=0.9)
        axes[0].fill_between(ts,counts,alpha=0.10,color=A1)
        if len(counts)>10:
            ww=max(3,len(counts)//20); roll=np.convolve(counts,np.ones(ww)/ww,"valid")
            rt=ts[ww//2:ww//2+len(roll)]
            axes[0].plot(rt,roll,color=A2,lw=2,ls="--",alpha=0.9,label="Smoothed avg")
            axes[0].legend(facecolor=PNL,edgecolor=GRD,labelcolor=TXT,fontsize=7)
        axes[0].set(xlabel="Time (s)",ylabel="Count",title="Count Over Time")
        vels=[]
        for d in all_tracks.values():
            pos=d["pos"]
            if len(pos)<2: continue
            dist=sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
            dur_t=(d["last"]-d["first"])/max(fps,1)
            if dur_t>0: vels.append(dist/dur_t)
        if vels:
            axes[1].hist(vels,bins=15,color=A1,edgecolor=BG,alpha=0.85)
            axes[1].axvline(np.mean(vels),color=A2,lw=2,ls="--",label=f"Avg:{np.mean(vels):.0f}")
            axes[1].legend(facecolor=PNL,edgecolor=GRD,labelcolor=TXT,fontsize=7)
        axes[1].set(xlabel="Speed (px/s)",ylabel="Count",title="Speed Distribution")
        clrs=plt.cm.tab10(np.linspace(0,1,max(1,len(all_tracks))))
        for idx,(tid,d) in enumerate(all_tracks.items()):
            pos=d["pos"]
            if len(pos)<2: continue
            xs=[p[0] for p in pos]; ys=[p[1] for p in pos]
            axes[2].plot(xs,ys,color=clrs[idx%len(clrs)],lw=0.8,alpha=0.7)
            axes[2].scatter(xs[0],ys[0],color=A2,s=12,zorder=5)
            axes[2].scatter(xs[-1],ys[-1],color="#dc2626",s=12,marker="x",lw=1.5,zorder=5)
        axes[2].set_xlim(0,W); axes[2].set_ylim(H,0)
        axes[2].set(xlabel="X",ylabel="Y",title="Paths  ● start  × end")
        axes[2].set_aspect("equal","datalim")
        sc=8; hm=np.zeros((H//sc+1,W//sc+1),dtype=np.float32)
        for d in all_tracks.values():
            for x,y,_ in d["pos"]:
                hm[min(int(y/sc),hm.shape[0]-1),min(int(x/sc),hm.shape[1]-1)]+=1
        from scipy.ndimage import gaussian_filter
        hm=gaussian_filter(hm,sigma=3)
        im=axes[3].imshow(hm,cmap="YlOrRd",aspect="auto")
        plt.colorbar(im,ax=axes[3],shrink=0.85).ax.tick_params(labelcolor=TXT,labelsize=7)
        axes[3].set(title="Heatmap",xlabel="X",ylabel="Y")
        fig.suptitle("Movement Analysis",fontsize=12,color=TXT,fontweight="bold",y=0.97)
        out=os.path.join(tempfile.gettempdir(),"analytics.png")
        plt.savefig(out,dpi=120,facecolor=BG,bbox_inches="tight"); plt.close()
        img=cv2.imread(out); return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Chart error: {e}"); return None

# ─────────────────────────────────────────────────────────
#  TRACKER + CANCEL FLAG
# ─────────────────────────────────────────────────────────
class Tracker:
    def __init__(self,max_gone=15,max_dist=80):
        self.nid=1; self.objs={}; self.bxs={}; self.gone={}
        self.mg=max_gone; self.md=max_dist
    def update(self,dets):
        if len(dets)==0:
            for oid in list(self.gone):
                self.gone[oid]+=1
                if self.gone[oid]>self.mg: del self.objs[oid],self.bxs[oid],self.gone[oid]
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
        D=cdist(oc,cens); rows=D.min(1).argsort(); cols=D.argmin(1)[rows]
        ur,uc,res=set(),set(),[]
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

class CancelFlag:
    def __init__(self):  self._stop=False
    def stop(self):      self._stop=True
    def reset(self):     self._stop=False
    def cancelled(self): return self._stop

CANCEL = CancelFlag()

# ─────────────────────────────────────────────────────────
#  HANDLER FUNCTIONS
# ─────────────────────────────────────────────────────────
def analyze_image(image,conf,speed):
    if image is None: return None,"Upload a photo, then click **Scan for Worms**.",None
    overlap=SPEED_OPTS[speed]["overlap"]; t0=time.time()
    boxes,scores=run_detection(image,conf,IOU_DEF,overlap); elapsed=time.time()-t0
    n=len(boxes); H,W=image.shape[:2]
    xs,ys=_steps(W,PATCH,overlap),_steps(H,PATCH,overlap)
    patches=(len(xs)*len(ys)) if (W>PATCH*1.5 or H>PATCH*1.5) else 1
    out=annotate(image,boxes,scores)
    out=stamp(out,f"{n} worms found",f"{elapsed:.1f}s · {patches} zones · {HW['dev']}")
    if n==0: return out,"No worms detected. Try lowering the sensitivity slider.",None
    avg_c=np.mean(scores); ww=boxes[:,2]-boxes[:,0]; hh=boxes[:,3]-boxes[:,1]
    info=(f"**{n} worms detected** in {elapsed:.1f}s\n\n"
          f"Avg confidence: **{avg_c:.0%}** · Avg size: **{np.mean(ww):.0f}×{np.mean(hh):.0f} px**")
    csv_p=os.path.join(tempfile.gettempdir(),"detections.csv")
    with open(csv_p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["worm_number","x1","y1","x2","y2","confidence","width_px","height_px"])
        for i,(b,s) in enumerate(zip(boxes,scores)):
            x1,y1,x2,y2=b
            w.writerow([i+1,f"{x1:.0f}",f"{y1:.0f}",f"{x2:.0f}",f"{y2:.0f}",f"{s:.3f}",f"{x2-x1:.0f}",f"{y2-y1:.0f}"])
    return out,info,csv_p

def start_tracking(video_path,conf,speed,trail_len,cx1,cy1,cx2,cy2,trim_s,trim_e,progress=gr.Progress()):
    CANCEL.reset()
    if not video_path: return None,"Upload a video first.",None,None
    cap=cv2.VideoCapture(video_path); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur=total/fps; cx1,cy1,cx2,cy2=int(cx1),int(cy1),int(cx2),int(cy2)
    use_crop=not(cx1==0 and cy1==0 and cx2==0 and cy2==0)
    if use_crop:
        cx1=max(0,min(cx1,ow-1)); cy1=max(0,min(cy1,oh-1))
        cx2=max(cx1+10,min(cx2,ow)); cy2=max(cy1+10,min(cy2,oh))
    else: cx1,cy1,cx2,cy2=0,0,ow,oh
    W,H=cx2-cx1,cy2-cy1; ts=max(0.0,float(trim_s)); te=float(trim_e)
    if te<=ts: te=dur
    sf=int(ts*fps); ef=min(int(te*fps),total); total_f=ef-sf
    preset=SPEED_OPTS[speed]; overlap=preset["overlap"]; skip=max(1,preset["skip"])
    to_proc=max(1,total_f//skip)
    out_p=os.path.join(tempfile.gettempdir(),"tracked.mp4")
    writer=cv2.VideoWriter(out_p,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    tracker=Tracker(max_gone=int(fps*2)); counts=[]; all_tracks={}; trails=defaultdict(list)
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
            tracked=tracker.update(boxes); tids,tboxes=[],[]
            for tid,box in tracked:
                tids.append(tid); tboxes.append(box)
                cx_p=(box[0]+box[2])/2; cy_p=(box[1]+box[3])/2
                if tid not in all_tracks: all_tracks[tid]={"first":fn,"pos":[]}
                all_tracks[tid]["last"]=fn; all_tracks[tid]["pos"].append((cx_p,cy_p,fn))
                trails[tid].append((cx_p,cy_p))
                if len(trails[tid])>trail_len: trails[tid]=trails[tid][-trail_len:]
            tboxes=np.array(tboxes) if tboxes else np.array([]).reshape(0,4)
            tscores=(scores[:len(tboxes)] if len(scores)>=len(tboxes) else np.ones(len(tboxes)))
            tids_a=np.array(tids) if tids else None; counts.append(len(tboxes))
            ann=annotate(frame,tboxes,tscores,tids_a,trails)
            ann=stamp(ann,f"{len(tboxes)} worms  ·  {len(all_tracks)} unique",
                      f"t={fn/fps:.1f}s  ·  ETA {fmt_time(eta_val)}")
            last_ann=ann; processed+=1
            if processed>1: eta_val=(to_proc-processed)*(time.time()-t0)/processed
        else: ann=last_ann if last_ann is not None else frame
        writer.write(ann)
        if fn%20==0: progress(fn/total_f,desc=f"Frame {fn}/{total_f}  ·  ETA {fmt_time(eta_val)}")
    cap.release(); writer.release()
    if CANCEL.cancelled(): return out_p,"⏹ Tracking stopped.",None,None
    csv_p=os.path.join(tempfile.gettempdir(),"tracking.csv")
    with open(csv_p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["worm_id","first_frame","last_frame","duration_s","total_distance_px","avg_speed_px_per_s","positions_recorded"])
        for tid,d in sorted(all_tracks.items()):
            pos=d["pos"]; first=d["first"]; last_f=d["last"]; dur_t=(last_f-first)/fps
            dist=sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
            w.writerow([tid,first,last_f,f"{dur_t:.2f}",f"{dist:.0f}",
                        f"{dist/dur_t:.1f}" if dur_t>0 else "0",len(pos)])
    total_t=time.time()-t0; vels,dists=[],[]
    for d in all_tracks.values():
        pos=d["pos"]
        if len(pos)<2: continue
        dist=sum(np.sqrt((pos[i][0]-pos[i-1][0])**2+(pos[i][1]-pos[i-1][1])**2) for i in range(1,len(pos)))
        dur_t=(d["last"]-d["first"])/fps; dists.append(dist)
        if dur_t>0: vels.append(dist/dur_t)
    avg_vel=f"{np.mean(vels):.0f} px/s" if vels else "—"
    avg_dist=f"{np.mean(dists):.0f} px" if dists else "—"
    info=(f"**Tracking complete** · {len(all_tracks)} unique worms · {fmt_time(total_t)}\n\n"
          f"| Metric | Value |\n|---|---|\n"
          f"| Video length | {total_f/fps:.1f}s @ {fps:.0f} fps |\n"
          f"| Avg worms/frame | {np.mean(counts):.1f} |\n"
          f"| Peak worms | {max(counts) if counts else 0} |\n"
          f"| Avg speed | {avg_vel} |\n| Avg distance | {avg_dist} |")
    chart=make_chart(all_tracks,counts,fps,W,H)
    return out_p,info,csv_p,chart

def stop_tracking(): CANCEL.stop(); return "⏹ Stop signal sent…"

def run_batch(files,conf,speed,progress=gr.Progress()):
    if not files: return None,"Upload some images first.",None
    overlap=SPEED_OPTS[speed]["overlap"]
    out_dir=os.path.join(tempfile.gettempdir(),"worm_batch")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir); rows,gallery=[],[]
    for i,fp in enumerate(files):
        progress((i+1)/len(files),desc=f"Image {i+1} of {len(files)}")
        img=cv2.imread(fp)
        if img is None: continue
        fname=os.path.basename(fp); boxes,scores=run_detection(img,conf,IOU_DEF,overlap)
        ann=annotate(img,boxes,scores); cv2.imwrite(os.path.join(out_dir,f"detected_{fname}"),ann)
        gallery.append((cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),f"{fname} — {len(boxes)} worms"))
        rows.append({"file":fname,"count":len(boxes),"avg_conf":f"{np.mean(scores):.0%}" if len(scores) else "—"})
    csv_p=os.path.join(out_dir,"summary.csv")
    with open(csv_p,"w",newline="") as f:
        dw=csv.DictWriter(f,["file","count","avg_conf"]); dw.writeheader(); dw.writerows(rows)
    zip_p=shutil.make_archive(os.path.join(tempfile.gettempdir(),"worm_batch"),"zip",out_dir)
    total=sum(r["count"] for r in rows); avg=total/len(rows) if rows else 0
    info=(f"**{len(rows)} images processed** · {total} worms · {avg:.1f} avg\n\n"
          "| File | Worms | Conf |\n|---|---|---|\n"+
          "\n".join(f"| {r['file']} | {r['count']} | {r['avg_conf']} |" for r in rows))
    return gallery,info,zip_p

def analyze_frame(video_path,conf,frame_no,cx1,cy1,cx2,cy2):
    if not video_path: return None,"Upload a video first."
    cap=cv2.VideoCapture(video_path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    ow=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fn=min(max(0,int(frame_no)),total-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES,fn); ok,frame=cap.read(); cap.release()
    if not ok: return None,"Cannot read that frame."
    cx1,cy1,cx2,cy2=int(cx1),int(cy1),int(cx2),int(cy2)
    if not(cx1==0 and cy1==0 and cx2==0 and cy2==0):
        cx1=max(0,min(cx1,ow-1)); cy1=max(0,min(cy1,oh-1))
        cx2=max(cx1+10,min(cx2,ow)); cy2=max(cy1+10,min(cy2,oh))
        frame=frame[cy1:cy2,cx1:cx2]
    boxes,scores=run_detection(frame,conf); ann=annotate(frame,boxes,scores)
    ann=stamp(ann,f"{len(boxes)} worms",f"frame {fn}  ·  {fn/fps:.2f}s")
    info=(f"**Frame {fn}** ({fn/fps:.2f}s) — **{len(boxes)} worms**" +
          (f" · avg conf {np.mean(scores):.0%}" if len(scores) else ""))
    return cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),info

def apply_trim_crop(path,x1,y1,x2,y2,ts,te,progress=gr.Progress()):
    if not path: return None,"Upload a video first.",None
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur=total/fps; x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    x1=max(0,min(x1,ow-1)); y1=max(0,min(y1,oh-1))
    x2=max(x1+10,min(x2,ow)); y2=max(y1+10,min(y2,oh))
    ts=max(0.0,float(ts)); te=float(te)
    if te<=ts: te=dur
    sf=int(ts*fps); ef=int(te*fps); nf=ef-sf; CW,CH=x2-x1,y2-y1
    out_p=os.path.join(tempfile.gettempdir(),"trimmed_cropped.mp4")
    wt=cv2.VideoWriter(out_p,cv2.VideoWriter_fourcc(*"mp4v"),fps,(CW,CH))
    cap.set(cv2.CAP_PROP_POS_FRAMES,sf); t0,done=time.time(),0
    for _ in range(nf):
        ok,frame=cap.read()
        if not ok: break
        wt.write(frame[y1:y2,x1:x2]); done+=1
        if done%100==0:
            el=time.time()-t0; rate=done/el if el>0 else 1
            progress(done/nf,desc=f"Frame {done}/{nf}  ·  ETA {fmt_time((nf-done)/rate)}")
    cap.release(); wt.release(); tt=time.time()-t0
    speedup=(ow*oh)/(CW*CH) if CW*CH else 1
    info=(f"**Done!** {CW}×{CH}px · {te-ts:.1f}s · {tt:.1f}s\n\n"
          f"Video is **{speedup:.1f}× smaller** — tracking will be proportionally faster.")
    return out_p,info,out_p


# ═══════════════════════════════════════════════════════════
#  CSS  ← key insight: style Gradio's .tab-nav directly
#         so there is only ever ONE nav bar, no custom HTML
#         nav needed, no JS tab-switching needed either.
# ═══════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:    #f1f5f9;
  --white: #ffffff;
  --b1:    #e2e8f0;
  --b2:    #cbd5e1;
  --text:  #0f172a;
  --muted: #64748b;
  --faint: #94a3b8;
  --blue:  #2563eb;
  --blue2: #1d4ed8;
  --bluel: #eff6ff;
  --green: #16a34a;
  --red:   #dc2626;
  --nh:    58px;        /* navbar height */
  --sbw:   300px;       /* sidebar width */
  --r:     10px;
  --sh:    0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04);
  --font:  'Inter', system-ui, sans-serif;
  --mono:  'JetBrains Mono', monospace;
}

/* ── Reset ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body {
  margin:0!important; padding:0!important;
  background:var(--bg)!important;
  font-family:var(--font)!important;
  color:var(--text)!important;
  -webkit-font-smoothing:antialiased;
}

/* ── Strip Gradio's container centering / max-width ─────── */
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .contain,
.contain, .app, #root {
  max-width:100%!important; width:100%!important;
  margin:0!important; padding:0!important;
  box-shadow:none!important; background:transparent!important;
}
.gradio-container { background:var(--bg)!important; }
.gap { gap:0!important; padding:0!important; margin:0!important; }
.tabs, .tabitem { background:var(--bg)!important; padding:0!important; margin:0!important; }

/* ── Strip block chrome ─────────────────────────────────── */
.block, .form, .panel {
  background:transparent!important; border:none!important;
  box-shadow:none!important; border-radius:0!important;
  padding:0!important; margin:0!important;
}

/* ════════════════════════════════════════════════════════
   THE NAVBAR — we style Gradio's real .tab-nav to look
   like a professional top bar. Zero JS needed. Zero
   duplicate navbars. One source of truth.
   ════════════════════════════════════════════════════════ */

/* Brand bar sits above the tab-nav, injected via gr.HTML */
#wa-brand-bar {
  position: sticky;
  top: 0;
  z-index: 9001;
  width: 100%;
  background: var(--white);
  border-bottom: none;
  display: flex;
  align-items: center;
  height: 52px;
  padding: 0 24px;
  gap: 12px;
}
#wa-icon {
  width:32px; height:32px; background:var(--blue);
  border-radius:8px; display:flex; align-items:center;
  justify-content:center; flex-shrink:0;
}
#wa-title { font-size:15px; font-weight:700; color:var(--text); letter-spacing:-0.3px; }
#wa-badge {
  font-family:var(--mono); font-size:10px; font-weight:500;
  color:var(--blue); background:var(--bluel);
  border:1px solid #bfdbfe; padding:2px 8px; border-radius:20px;
}
#wa-spacer { flex:1; }
#wa-hw {
  font-family:var(--mono); font-size:11px; color:var(--faint);
  padding-left:16px; border-left:1px solid var(--b1); white-space:nowrap;
}
#wa-hw b { color:var(--blue); font-weight:600; }

/* Style Gradio's .tab-nav as the tab strip beneath brand bar */
.tab-nav {
  position: sticky!important;
  top: 52px!important;           /* sit right below brand bar */
  z-index: 9000!important;
  width: 100%!important;
  background: var(--white)!important;
  border-bottom: 1px solid var(--b1)!important;
  border-top: 1px solid var(--b1)!important;
  display: flex!important;
  align-items: stretch!important;
  height: 44px!important;
  padding: 0 24px!important;
  gap: 0!important;
  margin: 0!important;
  overflow-x: auto!important;
  scrollbar-width: none!important;
  /* Remove Gradio's default styling */
  box-shadow: none!important;
  border-radius: 0!important;
}
.tab-nav::-webkit-scrollbar { display:none; }

/* Gradio tab buttons inside .tab-nav */
.tab-nav button {
  background: transparent!important;
  border: none!important;
  border-bottom: 2.5px solid transparent!important;
  color: var(--muted)!important;
  font-family: var(--font)!important;
  font-size: 13px!important;
  font-weight: 500!important;
  padding: 0 18px!important;
  height: 44px!important;
  cursor: pointer!important;
  white-space: nowrap!important;
  transition: color .15s, border-color .15s!important;
  border-radius: 0!important;
  outline: none!important;
  /* Remove Gradio's underline default */
  border-bottom-color: transparent!important;
  text-decoration: none!important;
}
.tab-nav button:hover {
  color: var(--text)!important;
  background: #f8fafc!important;
}
.tab-nav button.selected {
  color: var(--blue)!important;
  border-bottom-color: var(--blue)!important;
  font-weight: 600!important;
  background: transparent!important;
}

/* ════════════════════════════════════════════════════════
   LAYOUT — sticky sidebar + scrollable content
   Both columns are children of gr.Row → direct flex children
   ════════════════════════════════════════════════════════ */

/* The outer Row of each Tab */
.tabitem > .contain > .form,
.tabitem > .form,
.tabitem > div > div {
  display: flex!important;
  flex-direction: row!important;
  flex-wrap: nowrap!important;
  gap: 0!important;
  padding: 0!important;
  margin: 0!important;
  width: 100%!important;
  min-height: calc(100vh - 96px)!important;  /* 52 brand + 44 tabs */
  align-items: stretch!important;
}

/* First column child = sidebar */
.tabitem > .contain > .form > div:first-child,
.tabitem > .form > div:first-child,
.tabitem > div > div > div:first-child {
  flex: 0 0 var(--sbw)!important;
  width: var(--sbw)!important;
  min-width: var(--sbw)!important;
  max-width: var(--sbw)!important;
  background: var(--white)!important;
  border-right: 1px solid var(--b1)!important;
  overflow-y: auto!important;
  position: sticky!important;
  top: 96px!important;  /* brand + tabs */
  height: calc(100vh - 96px)!important;
  padding: 0!important;
}

/* Last column child = content */
.tabitem > .contain > .form > div:last-child,
.tabitem > .form > div:last-child,
.tabitem > div > div > div:last-child {
  flex: 1 1 0!important;
  min-width: 0!important;
  background: var(--bg)!important;
  padding: 24px!important;
  overflow-y: auto!important;
}

/* ════════════════════════════════════════════════════════
   SIDEBAR SECTION CARDS
   We use gr.HTML cards for structured sections
   ════════════════════════════════════════════════════════ */
.sbc {                          /* sidebar card */
  border-bottom: 1px solid var(--b1);
}
.sbc-h {                        /* header */
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 11px 16px 10px;
  background: #f8fafc;
  border-bottom: 1px solid var(--b1);
}
.sbc-icon {
  width: 20px; height: 20px; border-radius: 5px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.sbc-label {
  font-size: 10.5px!important; font-weight: 700!important;
  text-transform: uppercase!important; letter-spacing: 0.9px!important;
  color: var(--muted)!important; font-family: var(--font)!important;
}
.sbc-b {                        /* body */
  padding: 14px 16px 16px;
}
.sbc-btn {                      /* action area at bottom */
  padding: 12px 16px 16px;
  border-top: 1px solid var(--b1);
  background: #fafbfc;
}

/* ════════════════════════════════════════════════════════
   RESULT CARDS (content area)
   ════════════════════════════════════════════════════════ */
.rc {
  background: var(--white)!important;
  border: 1px solid var(--b1)!important;
  border-radius: var(--r)!important;
  box-shadow: var(--sh)!important;
  overflow: hidden!important;
  margin-bottom: 16px!important;
}
.rc-h {
  display: flex; align-items: center; gap: 8px;
  padding: 11px 16px;
  border-bottom: 1px solid var(--b1);
  background: #fafbfc;
}
.rc-title { font-size:12px; font-weight:600; color:var(--text); }
.rc-badge {
  font-family:var(--mono); font-size:10px; color:var(--muted);
  background:var(--bg); border:1px solid var(--b1);
  padding:1px 7px; border-radius:20px; margin-left:auto;
}
.rc-b { padding:16px; }

/* ════════════════════════════════════════════════════════
   BUTTONS
   ════════════════════════════════════════════════════════ */
button.primary, .gradio-button.primary {
  background:var(--blue)!important; color:#fff!important;
  border:none!important; border-radius:8px!important;
  font-family:var(--font)!important; font-size:13.5px!important;
  font-weight:600!important; padding:11px 20px!important;
  width:100%!important; cursor:pointer!important;
  box-shadow:0 1px 3px rgba(37,99,235,.25)!important;
  transition:background .15s,box-shadow .15s,transform .1s!important;
}
button.primary:hover {
  background:var(--blue2)!important;
  box-shadow:0 4px 12px rgba(37,99,235,.30)!important;
  transform:translateY(-1px)!important;
}
button.primary:active { transform:translateY(0)!important; }

/* Stop button */
#stop-btn > button, #stop-btn button {
  background:#fff5f5!important; color:var(--red)!important;
  border:1px solid #fca5a5!important; border-radius:8px!important;
  font-family:var(--font)!important; font-size:13.5px!important;
  font-weight:600!important; padding:10px 20px!important;
  width:100%!important; cursor:pointer!important; margin-top:8px!important;
  transition:background .15s,border-color .15s!important;
}
#stop-btn > button:hover { background:#fee2e2!important; border-color:var(--red)!important; }

/* ════════════════════════════════════════════════════════
   FORM CONTROLS
   ════════════════════════════════════════════════════════ */
label > span:first-child, .label-wrap > span {
  font-size:12px!important; font-weight:500!important;
  color:var(--muted)!important; font-family:var(--font)!important;
}
input[type=number], input[type=text], textarea {
  background:var(--bg)!important; border:1px solid var(--b2)!important;
  border-radius:7px!important; color:var(--text)!important;
  font-size:13px!important; padding:8px 10px!important;
  transition:border-color .15s,box-shadow .15s!important;
}
input:focus, textarea:focus {
  border-color:var(--blue)!important;
  box-shadow:0 0 0 3px rgba(37,99,235,.10)!important; outline:none!important;
}
input[type=range]    { accent-color:var(--blue)!important; }
input[type=checkbox] { accent-color:var(--blue)!important; }
input[type=radio]    { accent-color:var(--blue)!important; }

/* ════════════════════════════════════════════════════════
   ACCORDION
   ════════════════════════════════════════════════════════ */
details {
  background:#f8fafc!important; border:1px solid var(--b1)!important;
  border-radius:8px!important; overflow:hidden!important; margin-top:8px!important;
}
details summary {
  padding:10px 14px!important; font-size:12.5px!important;
  font-weight:500!important; color:var(--muted)!important; cursor:pointer!important;
}
details summary:hover { color:var(--text)!important; }
details[open] summary { color:var(--blue)!important; }

/* ════════════════════════════════════════════════════════
   MARKDOWN
   ════════════════════════════════════════════════════════ */
.prose, .md, .markdown {
  font-family:var(--font)!important; font-size:13.5px!important;
  line-height:1.7!important; color:var(--text)!important;
}
.prose h2, .md h2 {
  font-size:17px!important; font-weight:700!important;
  margin:0 0 14px!important; padding-bottom:10px!important;
  border-bottom:1px solid var(--b1)!important; letter-spacing:-0.3px!important;
}
.prose h3, .md h3 {
  font-size:11px!important; font-weight:700!important;
  color:var(--faint)!important; text-transform:uppercase!important;
  letter-spacing:0.8px!important; margin:18px 0 8px!important;
}
.prose strong, .md strong { color:var(--blue)!important; font-weight:600!important; }
.prose table, .md table {
  width:100%!important; border-collapse:collapse!important;
  font-size:13px!important; margin-top:10px!important;
  border:1px solid var(--b1)!important; border-radius:8px!important; overflow:hidden!important;
}
.prose th, .md th {
  background:var(--bg)!important; padding:8px 13px!important;
  font-size:10px!important; font-weight:700!important; color:var(--faint)!important;
  text-align:left!important; text-transform:uppercase!important;
  letter-spacing:0.5px!important; border-bottom:1px solid var(--b1)!important;
}
.prose td, .md td { padding:8px 13px!important; border-bottom:1px solid var(--b1)!important; }
.prose tr:last-child td, .md tr:last-child td { border-bottom:none!important; }

/* ════════════════════════════════════════════════════════
   MISC
   ════════════════════════════════════════════════════════ */
.gradio-container img { border-radius:8px!important; }
footer, footer.svelte-mpyp5e { display:none!important; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--b2); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--faint); }
"""

# ─────────────────────────────────────────────────────────
#  HTML HELPERS
# ─────────────────────────────────────────────────────────
def sbc(title, icon_bg="#eff6ff", icon_color="#2563eb", icon_path="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"):
    """Open a sidebar card with colored icon header."""
    return gr.HTML(f"""<div class="sbc"><div class="sbc-h">
      <div class="sbc-icon" style="background:{icon_bg}">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="{icon_color}"><path d="{icon_path}"/></svg>
      </div>
      <span class="sbc-label">{title}</span>
    </div><div class="sbc-b">""")

def sbc_end():
    return gr.HTML("</div></div>")

def rc_open(title, badge=""):
    b = f'<span class="rc-badge">{badge}</span>' if badge else ""
    return gr.HTML(f'<div class="rc"><div class="rc-h"><span class="rc-title">{title}</span>{b}</div><div class="rc-b">')

def rc_close():
    return gr.HTML("</div></div>")


# ═══════════════════════════════════════════════════════════
#  BUILD UI
# ═══════════════════════════════════════════════════════════
with gr.Blocks(title="Worm Analyzer", css=CSS) as app:

    # ── Brand bar (above the styled Gradio tab nav) ─────────
    gr.HTML(f"""
    <div id="wa-brand-bar">
      <div id="wa-icon">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="white" stroke-width="2.5" stroke-linecap="round">
          <path d="M3 12c0-3 2-5 4-5s3 2 5 2 3-2 5-2 4 2 4 5-2 5-4 5-3-2-5-2-3 2-5 2-4-2-4-5z"/>
        </svg>
      </div>
      <span id="wa-title">Worm Analyzer</span>
      <span id="wa-badge">YOLOv8m</span>
      <div id="wa-spacer"></div>
      <div id="wa-hw">{HW['dev']} &nbsp;·&nbsp; <b>{HW['ms']:.0f}ms</b>/scan</div>
    </div>
    """)

    # Gradio renders .tab-nav here — now styled to look like our tab strip
    with gr.Tabs():

        # ══════════════════════════════════════
        # TAB 1 — Scan Photo
        # ══════════════════════════════════════
        with gr.Tab("Scan Photo"):
            with gr.Row():
                # ── SIDEBAR ──────────────────────────────────
                with gr.Column(scale=3, min_width=300):

                    sbc("Input Image", "#eff6ff", "#2563eb",
                        "M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z")
                    img_in = gr.Image(type="numpy", show_label=False, height=210)
                    sbc_end()

                    sbc("Sensitivity", "#f0fdf4", "#16a34a",
                        "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z")
                    img_conf = gr.Slider(0.10, 0.90, CONF_DEF, step=0.05,
                        label="Threshold",
                        info="Lower = more detections · Higher = fewer false positives")
                    sbc_end()

                    sbc("Scan Quality", "#faf5ff", "#7c3aed",
                        "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z")
                    img_speed = gr.Radio(list(SPEED_OPTS.keys()), value=DEF_SPEED, show_label=False)
                    sbc_end()

                    gr.HTML('<div class="sbc-btn">')
                    img_btn = gr.Button("🔍  Scan for Worms", variant="primary", size="lg")
                    gr.HTML('</div>')

                # ── CONTENT ──────────────────────────────────
                with gr.Column(scale=7, min_width=0):
                    rc_open("Detection Result", "Ready")
                    img_out  = gr.Image(show_label=False, height=460, container=False)
                    img_info = gr.Markdown("Upload a photo on the left, then click **Scan for Worms**.")
                    img_csv  = gr.File(label="📄 Download detections CSV")
                    rc_close()

            img_btn.click(analyze_image, [img_in, img_conf, img_speed],
                          [img_out, img_info, img_csv])

        # ══════════════════════════════════════
        # TAB 2 — Track Video
        # ══════════════════════════════════════
        with gr.Tab("Track Video"):
            vt_dd = gr.State(); vt_frm = gr.State()

            with gr.Row():
                with gr.Column(scale=3, min_width=300):

                    sbc("Video File", "#eff6ff", "#2563eb",
                        "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z")
                    vt_vid  = gr.Video(show_label=False, height=160)
                    vt_info = gr.Markdown("Upload a video to get started.")
                    sbc_end()

                    sbc("Scan Region", "#f0fdf4", "#16a34a",
                        "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z")
                    with gr.Accordion("Dish crop settings", open=True):
                        with gr.Row():
                            vt_auto = gr.Checkbox(label="Auto-detect dish", value=True)
                            vt_pad  = gr.Slider(-50, 150, 50, step=5, label="Padding (px)")
                        vt_prev = gr.Image(show_label=False, height=110, container=False)
                        vt_x1 = gr.Number(0, visible=False, precision=0)
                        vt_y1 = gr.Number(0, visible=False, precision=0)
                        vt_x2 = gr.Number(0, visible=False, precision=0)
                        vt_y2 = gr.Number(0, visible=False, precision=0)
                    sbc_end()

                    sbc("Time Range & Quality", "#fff7ed", "#d97706",
                        "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z")
                    with gr.Row():
                        vt_ts = gr.Number(0, label="Start (s)", precision=1, minimum=0)
                        vt_te = gr.Number(0, label="End (0=full)", precision=1, minimum=0)
                    vt_speed = gr.Radio(list(SPEED_OPTS.keys()), value=DEF_SPEED, label="Quality")
                    vt_conf  = gr.Slider(0.10, 0.90, CONF_DEF, step=0.05, label="Sensitivity")
                    vt_trail = gr.Slider(10, 200, 60, step=10, label="Trail length (frames)")
                    sbc_end()

                    gr.HTML('<div class="sbc-btn">')
                    vt_btn  = gr.Button("▶  Start Tracking", variant="primary")
                    vt_stop = gr.Button("⏹  Stop Tracking", variant="secondary", elem_id="stop-btn")
                    vt_msg  = gr.Markdown("")
                    gr.HTML('</div>')

                with gr.Column(scale=7, min_width=0):
                    rc_open("Tracked Video", "Results appear here")
                    vt_out  = gr.Video(show_label=False, height=340)
                    vt_stat = gr.Markdown("Start tracking to see results.")
                    rc_close()

                    rc_open("Movement Analytics")
                    vt_chart = gr.Image(show_label=False, height=260, container=False)
                    vt_csv   = gr.File(label="📄 Download tracking CSV")
                    rc_close()

            def _vt_up(p):
                dd,rgb,x1,y1,x2,y2,info,dur = load_video(p)
                prev = make_preview(dd,rgb,x1,y1,x2,y2) if dd else None
                return dd,rgb,x1,y1,x2,y2,info,prev,0,max(dur,0)
            def _vt_cr(dd,rgb,a,p):
                x1,y1,x2,y2,prev=update_crop(dd,rgb,a,p); return x1,y1,x2,y2,prev

            vt_vid.change(_vt_up,[vt_vid],
                          [vt_dd,vt_frm,vt_x1,vt_y1,vt_x2,vt_y2,vt_info,vt_prev,vt_ts,vt_te])
            vt_auto.change(_vt_cr,[vt_dd,vt_frm,vt_auto,vt_pad],[vt_x1,vt_y1,vt_x2,vt_y2,vt_prev])
            vt_pad.change( _vt_cr,[vt_dd,vt_frm,vt_auto,vt_pad],[vt_x1,vt_y1,vt_x2,vt_y2,vt_prev])
            vt_btn.click(start_tracking,
                [vt_vid,vt_conf,vt_speed,vt_trail,vt_x1,vt_y1,vt_x2,vt_y2,vt_ts,vt_te],
                [vt_out,vt_stat,vt_csv,vt_chart])
            vt_stop.click(stop_tracking,[],[vt_msg])

        # ══════════════════════════════════════
        # TAB 3 — Batch Scan
        # ══════════════════════════════════════
        with gr.Tab("Batch Scan"):
            with gr.Row():
                with gr.Column(scale=3, min_width=300):

                    sbc("Select Images", "#eff6ff", "#2563eb",
                        "M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2z")
                    bt_files = gr.File(file_count="multiple", file_types=["image"],
                                       show_label=False)
                    sbc_end()

                    sbc("Settings", "#faf5ff", "#7c3aed",
                        "M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z")
                    bt_speed = gr.Radio(list(SPEED_OPTS.keys()), value=DEF_SPEED, label="Quality")
                    bt_conf  = gr.Slider(0.10, 0.90, CONF_DEF, step=0.05, label="Sensitivity")
                    sbc_end()

                    gr.HTML('<div class="sbc-btn">')
                    bt_btn  = gr.Button("🔍  Scan All Images", variant="primary", size="lg")
                    bt_info = gr.Markdown("Upload images then click Scan All.")
                    bt_zip  = gr.File(label="📦 Download ZIP")
                    gr.HTML('</div>')

                with gr.Column(scale=7, min_width=0):
                    rc_open("Annotated Results")
                    bt_gallery = gr.Gallery(show_label=False, columns=3,
                                            height=600, object_fit="contain")
                    rc_close()

            bt_btn.click(run_batch,[bt_files,bt_conf,bt_speed],[bt_gallery,bt_info,bt_zip])

        # ══════════════════════════════════════
        # TAB 4 — Frame Explorer
        # ══════════════════════════════════════
        with gr.Tab("Frame Explorer"):
            fe_dd = gr.State(); fe_fst = gr.State()

            with gr.Row():
                with gr.Column(scale=3, min_width=300):

                    sbc("Video", "#eff6ff", "#2563eb",
                        "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z")
                    fe_vid = gr.Video(show_label=False, height=160)
                    sbc_end()

                    sbc("Navigation", "#fff7ed", "#d97706",
                        "M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z")
                    fe_frame = gr.Slider(0, 1000, 0, step=1, label="Frame number",
                                         info="Drag to scrub through the video")
                    fe_conf  = gr.Slider(0.10, 0.90, CONF_DEF, step=0.05, label="Sensitivity")
                    with gr.Accordion("Dish crop", open=False):
                        fe_msg = gr.Markdown("Upload a video to enable.")
                        with gr.Row():
                            fe_auto = gr.Checkbox(label="Auto-crop", value=True)
                            fe_pad  = gr.Slider(-50, 150, 50, step=5, label="Padding (px)")
                        fe_x1 = gr.Number(0, visible=False, precision=0)
                        fe_y1 = gr.Number(0, visible=False, precision=0)
                        fe_x2 = gr.Number(0, visible=False, precision=0)
                        fe_y2 = gr.Number(0, visible=False, precision=0)
                    sbc_end()

                    gr.HTML('<div class="sbc-btn">')
                    fe_btn = gr.Button("🔍  Analyze Frame", variant="primary", size="lg")
                    gr.HTML('</div>')

                with gr.Column(scale=7, min_width=0):
                    rc_open("Frame Analysis")
                    fe_out  = gr.Image(show_label=False, height=500, container=False)
                    fe_info = gr.Markdown("Choose a frame, then click **Analyze Frame**.")
                    rc_close()

            def _fe_up(p):
                dd,rgb,x1,y1,x2,y2,info,_ = load_video(p)
                cap=cv2.VideoCapture(p) if p else None
                total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 1
                if cap: cap.release()
                return gr.Slider(maximum=max(1,total-1)),dd,rgb,info,x1,y1,x2,y2
            def _fe_cr(dd,rgb,a,p):
                x1,y1,x2,y2,_=update_crop(dd,rgb,a,p); return x1,y1,x2,y2

            fe_vid.change(_fe_up,[fe_vid],[fe_frame,fe_dd,fe_fst,fe_msg,fe_x1,fe_y1,fe_x2,fe_y2])
            fe_auto.change(_fe_cr,[fe_dd,fe_fst,fe_auto,fe_pad],[fe_x1,fe_y1,fe_x2,fe_y2])
            fe_pad.change( _fe_cr,[fe_dd,fe_fst,fe_auto,fe_pad],[fe_x1,fe_y1,fe_x2,fe_y2])
            fe_btn.click(analyze_frame,[fe_vid,fe_conf,fe_frame,fe_x1,fe_y1,fe_x2,fe_y2],[fe_out,fe_info])

        # ══════════════════════════════════════
        # TAB 5 — Prepare Video
        # ══════════════════════════════════════
        with gr.Tab("Prepare Video"):
            vl_dd = gr.State(); vl_frm = gr.State()

            with gr.Row():
                with gr.Column(scale=3, min_width=300):

                    sbc("Source Video", "#eff6ff", "#2563eb",
                        "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z")
                    vl_vid  = gr.Video(show_label=False, height=160)
                    vl_info = gr.Markdown("Upload a video to begin.")
                    sbc_end()

                    sbc("Crop Region", "#f0fdf4", "#16a34a",
                        "M17 15h2V7h-8v2h6v6zm-2 2H5V9h6V7H5c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2v-4h-2v4z")
                    with gr.Row():
                        vl_auto = gr.Checkbox(label="Auto-detect dish", value=True)
                        vl_pad  = gr.Slider(-50, 150, 50, step=5, label="Padding (px)")
                    vl_x1 = gr.Number(0,    visible=False, precision=0)
                    vl_y1 = gr.Number(0,    visible=False, precision=0)
                    vl_x2 = gr.Number(1920, visible=False, precision=0)
                    vl_y2 = gr.Number(1080, visible=False, precision=0)
                    sbc_end()

                    sbc("Trim", "#fff7ed", "#d97706",
                        "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z")
                    with gr.Row():
                        vl_ts = gr.Number(0, label="Start (s)", precision=1, minimum=0)
                        vl_te = gr.Number(0, label="End (0=all)", precision=1, minimum=0)
                    sbc_end()

                    gr.HTML('<div class="sbc-btn">')
                    vl_btn = gr.Button("✂  Apply Crop & Trim", variant="primary", size="lg")
                    gr.HTML('</div>')

                with gr.Column(scale=7, min_width=0):
                    rc_open("Preview & Output")
                    with gr.Tabs():
                        with gr.Tab("Preview — area to keep"):
                            vl_prev = gr.Image(show_label=False, height=360, container=False)
                        with gr.Tab("Processed video"):
                            vl_out  = gr.Video(show_label=False, height=360)
                    vl_stat = gr.Markdown("Set crop & trim, then click **Apply**.")
                    vl_dl   = gr.File(label="📄 Download processed video")
                    rc_close()

            def _vl_up(p):
                dd,rgb,x1,y1,x2,y2,info,dur = load_video(p)
                prev = make_preview(dd,rgb,x1,y1,x2,y2) if dd else None
                return dd,rgb,info,x1,y1,x2,y2,0,max(dur,0),prev
            def _vl_cr(dd,rgb,a,p):
                x1,y1,x2,y2,prev=update_crop(dd,rgb,a,p); return x1,y1,x2,y2,prev

            vl_vid.change(_vl_up,[vl_vid],
                [vl_dd,vl_frm,vl_info,vl_x1,vl_y1,vl_x2,vl_y2,vl_ts,vl_te,vl_prev])
            vl_auto.change(_vl_cr,[vl_dd,vl_frm,vl_auto,vl_pad],[vl_x1,vl_y1,vl_x2,vl_y2,vl_prev])
            vl_pad.change( _vl_cr,[vl_dd,vl_frm,vl_auto,vl_pad],[vl_x1,vl_y1,vl_x2,vl_y2,vl_prev])
            vl_btn.click(apply_trim_crop,
                [vl_vid,vl_x1,vl_y1,vl_x2,vl_y2,vl_ts,vl_te],[vl_out,vl_stat,vl_dl])

        # ══════════════════════════════════════
        # TAB 6 — Guide
        # ══════════════════════════════════════
        with gr.Tab("Guide"):
            with gr.Row():
                with gr.Column(scale=10, min_width=0):
                    gr.HTML(f"""
<div style="padding:28px;max-width:960px;margin:0 auto;">

  <!-- Stats row -->
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:24px;">
    <div class="rc" style="text-align:center;padding:20px;margin:0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:#2563eb;line-height:1.1">{HW['ms']:.0f}ms</div>
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-top:6px">per scan zone</div>
    </div>
    <div class="rc" style="text-align:center;padding:20px;margin:0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:#16a34a;line-height:1.1">25.8M</div>
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-top:6px">model parameters</div>
    </div>
    <div class="rc" style="text-align:center;padding:20px;margin:0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:#d97706;line-height:1.1">{HW['dev'].split()[0]}</div>
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-top:6px">compute device</div>
    </div>
  </div>

  <!-- Two-column cards -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
    <div class="rc">
      <div class="rc-h"><span class="rc-title">Feature Overview</span></div>
      <div class="rc-b">
    """)
                    gr.Markdown("""
**Scan Photo** — Upload a microscope image. Each worm is marked with bracket corners and a confidence score. Download coordinates as CSV.

**Track Video** — Centroid-based tracking assigns persistent IDs and draws movement trails. Sliding-window detection handles any frame size. Click Stop anytime for partial results.

**Batch Scan** — Process entire folders at once. Downloads as a ZIP with annotated images and a summary spreadsheet.

**Frame Explorer** — Step through individual frames to calibrate sensitivity before a full tracking run.

**Prepare Video** — Trim and crop before tracking. A 2× smaller frame = ~4× faster processing.
""")
                    gr.HTML(f"""
      </div>
    </div>
    <div class="rc">
      <div class="rc-h"><span class="rc-title">Settings Reference</span></div>
      <div class="rc-b">
    """)
                    gr.Markdown(f"""
### Sensitivity

| Value | Result |
|---|---|
| 0.20 – 0.35 | Finds faint worms · some false positives |
| 0.35 – 0.55 | Balanced · recommended for most experiments |
| 0.55 – 0.80 | High confidence only · minimal false positives |

### Your Hardware

| | |
|---|---|
| Device | {HW['dev']} |
| Name | {HW['name']} |
| Speed | {HW['ms']:.0f}ms / 416×416 zone |
| Default | {DEF_SPEED.split("—")[0].strip()} |
| Mode | {'GPU batch ×8' if 'GPU' in HW['dev'] else 'CPU sequential'} |
""")
                    gr.HTML("</div></div></div></div>")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Worm Analyzer  →  http://localhost:7860\n")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)