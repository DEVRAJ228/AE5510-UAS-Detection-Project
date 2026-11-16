# Source-derived detector training workflows from uas-yolo.
# Contains alternative configurations; inspect cell headings before running.


# ========================================================================
# Detector training – source cell 14
# ========================================================================
# ===== CELL 13: CLEAN LABELS + EXPERIMENT TOOLKIT =====
import os, re, json, time, random, shutil, collections

OUT = "/kaggle/working/yolo_data"
NAMES = {0: "Aircraft", 1: "Building", 2: "Military Truck", 3: "Tank"}

def clean_label_file(path):
    rows, changed = [], False
    for line in open(path):
        p = line.split()
        if not p:
            continue
        try:
            c = int(float(p[0])); v = [float(x) for x in p[1:]]
        except ValueError:
            changed = True; continue
        if len(v) == 4:
            x1, y1, x2, y2 = v[0]-v[2]/2, v[1]-v[3]/2, v[0]+v[2]/2, v[1]+v[3]/2
        elif len(v) >= 6 and len(v) % 2 == 0:          # polygon row -> bounding box
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            changed = True
        else:
            changed = True; continue
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)   # clip to image
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            changed = True
        w, h = cx2 - cx1, cy2 - cy1
        if w < 0.002 or h < 0.002:
            changed = True; continue
        rows.append(f"{c} {(cx1+cx2)/2:.6f} {(cy1+cy2)/2:.6f} {w:.6f} {h:.6f}")
    return rows, changed

for split in ["train", "val"]:                          # test labels stay untouched
    d = f"{OUT}/labels/{split}"
    fixed = 0
    for f in os.listdir(d):
        rows, changed = clean_label_file(f"{d}/{f}")
        if changed:
            fixed += 1
            with open(f"{d}/{f}", "w") as fh:
                fh.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{split}: fixed {fixed} of {len(os.listdir(d))} label files")

def source_key(fname):
    m = re.match(r'^(.*?\.rf\.[0-9a-f]+)', fname)
    return m.group(1) if m else os.path.splitext(fname)[0]

def aug_type(fname):
    rest = os.path.splitext(fname)[0][len(source_key(fname)):]
    return re.sub(r'_\d+$', '', rest).lstrip('_') or "original"

ALL_TYPES = ["flip", "brightness", "haze", "shadow", "motionblur", "zoom_in", "zoom_out", "rotation"]
HELD_PATH = "/kaggle/working/held_photos.json"

def build_split(keep_types, per_type, val_fraction=0.20):
    SPLIT = "/kaggle/working/yolo_split"
    photos = collections.defaultdict(list)
    for f in sorted(os.listdir(f"{OUT}/images/train")):
        photos[source_key(f)].append(f)

    if os.path.exists(HELD_PATH):                       # same held-out photos for every experiment
        held = set(json.load(open(HELD_PATH)))
    else:
        dom = {}
        for k, files in photos.items():
            cnt = collections.Counter()
            stem = os.path.splitext(files[0])[0]
            for line in open(f"{OUT}/labels/train/{stem}.txt"):
                p = line.split()
                if len(p) >= 5:
                    cnt[int(float(p[0]))] += 1
            dom[k] = cnt.most_common(1)[0][0] if cnt else -1
        rng, held = random.Random(42), set()
        for c in sorted(set(dom.values())):
            keys = sorted(k for k, v in dom.items() if v == c)
            rng.shuffle(keys)
            held.update(keys[:max(1, round(val_fraction * len(keys)))])
        json.dump(sorted(held), open(HELD_PATH, "w"))

    if os.path.exists(SPLIT):
        shutil.rmtree(SPLIT)
    for s in ["train", "val", "test"]:
        os.makedirs(f"{SPLIT}/images/{s}"); os.makedirs(f"{SPLIT}/labels/{s}")

    def put(src_split, dst_split, fname):
        stem = os.path.splitext(fname)[0]
        for kind, name in (("images", fname), ("labels", stem + ".txt")):
            src, dst = f"{OUT}/{kind}/{src_split}/{name}", f"{SPLIT}/{kind}/{dst_split}/{name}"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy(src, dst)

    for k, files in photos.items():
        if k in held:
            continue
        taken = collections.Counter()
        for f in files:
            t = aug_type(f)
            if t in keep_types and taken[t] < per_type:
                put("train", "train", f); taken[t] += 1
    for f in os.listdir(f"{OUT}/images/val"):
        put("val", "val", f)
    for k in held:
        flips = [f for f in photos[k] if aug_type(f) == "flip"]
        put("train", "val", (flips or photos[k])[0])
    for f in os.listdir(f"{OUT}/images/test"):
        put("test", "test", f)

    yaml_path = "/kaggle/working/data_split.yaml"
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {SPLIT}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                 "names:\n  0: Aircraft\n  1: Building\n  2: Military Truck\n  3: Tank\n")
    for s in ["train", "val", "test"]:
        cnt = collections.Counter()
        for f in os.listdir(f"{SPLIT}/labels/{s}"):
            for line in open(f"{SPLIT}/labels/{s}/{f}"):
                p = line.split()
                if len(p) >= 5:
                    cnt[int(float(p[0]))] += 1
        n_img = len(os.listdir(f"{SPLIT}/images/{s}"))
        print(f"  {s}: {n_img} images | " + ", ".join(f"{NAMES[c]} {cnt[c]}" for c in NAMES))
    return yaml_path

RESULTS_PATH = "/kaggle/working/results.json"
results_log = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
results_log.setdefault("exp1_ref", {"minutes": 26,
    "val": {"P": 0.510, "R": 0.320, "map50": 0.291, "map5095": 0.170, "ap50": {}},
    "test": {"P": 0.640, "R": 0.360, "map50": 0.433, "map5095": 0.226,
             "ap50": {"Aircraft": 0.628, "Building": 0.633, "Military Truck": 0.011, "Tank": 0.458}}})

def run_experiment(cfg):
    from ultralytics import YOLO
    t0 = time.time()
    print("\n" + "=" * 70); print("EXPERIMENT", cfg["name"], cfg); print("=" * 70)
    yaml_path = build_split(cfg["keep"], cfg["per_type"])
    YOLO(cfg["model"]).train(
        data=yaml_path, epochs=cfg["epochs"], patience=30, imgsz=cfg["imgsz"], batch=cfg["batch"],
        device=0, workers=2, project="/kaggle/working/runs", name=cfg["name"], exist_ok=True,
        seed=0, flipud=0.5, cos_lr=True, plots=False)
    best_path = f"/kaggle/working/runs/{cfg['name']}/weights/best.pt"
    best = YOLO(best_path)
    out = {"minutes": round((time.time() - t0) / 60, 1)}
    for split in ["val", "test"]:
        r = best.val(data=yaml_path, split=split, imgsz=cfg["imgsz"], device=0, plots=False, verbose=False)
        out[split] = {"P": float(r.box.mp), "R": float(r.box.mr), "map50": float(r.box.map50),
                      "map5095": float(r.box.map),
                      "ap50": {best.names[int(c)]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}}
    shutil.copy(best_path, f"/kaggle/working/{cfg['name']}_best.pt")
    results_log[cfg["name"]] = out
    json.dump(results_log, open(RESULTS_PATH, "w"), indent=1)
    return out

def show_summary():
    head = f"{'experiment':15s}|{'VAL mAP50':>10s} |{'TEST P':>7s}{'R':>6s}{'mAP50':>7s}{'mAP5095':>8s} |"
    head += "".join(f"{NAMES[c][:8]:>9s}" for c in NAMES) + " | min"
    print(head)
    for name, o in results_log.items():
        t = o["test"]
        row = f"{name:15s}|{o['val']['map50']:10.3f} |{t['P']:7.3f}{t['R']:6.3f}{t['map50']:7.3f}{t['map5095']:8.3f} |"
        row += "".join(f"{t['ap50'].get(NAMES[c], 0):9.3f}" for c in NAMES) + f" | {o['minutes']}"
        print(row)
print("Toolkit ready.")


# ========================================================================
# Detector training – source cell 15
# ========================================================================
CONFIGS = [
    dict(name="e2_s640",  model="yolov8s.pt", imgsz=640,  batch=16, epochs=100, keep=ALL_TYPES, per_type=1),
    dict(name="e3_s1024", model="yolov8s.pt", imgsz=1024, batch=8,  epochs=100, keep=ALL_TYPES, per_type=1),
    # only if e3 beats e2 (about 70+ min):
    # dict(name="e4_m1024", model="yolov8m.pt", imgsz=1024, batch=8, epochs=100, keep=ALL_TYPES, per_type=1),
]
for cfg in CONFIGS:
    if cfg["name"] in results_log:
        print("skip (already done):", cfg["name"]); continue
    try:
        run_experiment(cfg)
    except Exception as e:
        print("FAILED", cfg["name"], repr(e))
show_summary()


# ========================================================================
# Detector training – source cell 16
# ========================================================================
# ===== CELL 15: TOOLKIT WITH EXTRA TRAINING OPTIONS =====
def run_experiment(cfg):
    from ultralytics import YOLO
    t0 = time.time()
    print("\n" + "=" * 70); print("EXPERIMENT", cfg["name"]); print(cfg); print("=" * 70)
    yaml_path = build_split(cfg["keep"], cfg["per_type"])
    args = dict(data=yaml_path, epochs=cfg["epochs"], patience=30, imgsz=cfg["imgsz"],
                batch=cfg["batch"], device=0, workers=2, project="/kaggle/working/runs",
                name=cfg["name"], exist_ok=True, seed=0, flipud=0.5, cos_lr=True, plots=False)
    args.update(cfg.get("extra", {}))
    YOLO(cfg["model"]).train(**args)
    best_path = f"/kaggle/working/runs/{cfg['name']}/weights/best.pt"
    best = YOLO(best_path)
    out = {"minutes": round((time.time() - t0) / 60, 1)}
    for split in ["val", "test"]:
        r = best.val(data=yaml_path, split=split, imgsz=cfg["imgsz"], device=0, plots=False, verbose=False)
        out[split] = {"P": float(r.box.mp), "R": float(r.box.mr), "map50": float(r.box.map50),
                      "map5095": float(r.box.map),
                      "ap50": {best.names[int(c)]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}}
    shutil.copy(best_path, f"/kaggle/working/{cfg['name']}_best.pt")
    results_log[cfg["name"]] = out
    json.dump(results_log, open(RESULTS_PATH, "w"), indent=1)
    return out
print("run_experiment upgraded")


# ========================================================================
# Detector training – source cell 17
# ========================================================================
CONFIGS = [
    dict(name="e4_s640_freeze", model="yolov8s.pt", imgsz=640, batch=16, epochs=100,
         keep=ALL_TYPES, per_type=1,
         extra=dict(freeze=10, lr0=0.001, optimizer="AdamW")),
    dict(name="e5_s640_aug", model="yolov8s.pt", imgsz=640, batch=16, epochs=100,
         keep=ALL_TYPES, per_type=1,
         extra=dict(mixup=0.15, scale=0.7, degrees=10, translate=0.2, hsv_s=0.8, hsv_v=0.5)),
    dict(name="e6_m640", model="yolov8m.pt", imgsz=640, batch=16, epochs=100,
         keep=ALL_TYPES, per_type=1),
]
for cfg in CONFIGS:
    if cfg["name"] in results_log:
        print("skip (already done):", cfg["name"]); continue
    try:
        run_experiment(cfg)
    except Exception as e:
        print("FAILED", cfg["name"], repr(e))
show_summary()


# ========================================================================
# Detector training – source cell 18
# ========================================================================

# ===== CELL 17: FINAL MODEL ON ALL PHOTOS =====
from ultralytics import YOLO

# Winner: e5_s640_aug
FINAL = dict(
    name="final",
    model="yolov8s.pt",
    imgsz=640,
    batch=16,
    epochs=40,
    keep=ALL_TYPES,
    per_type=1,
    extra=dict(
        mixup=0.15,
        scale=0.7,
        degrees=10,
        translate=0.2,
        hsv_s=0.8,
        hsv_v=0.5
    )
)

def build_final_split(keep_types, per_type):
    SPLIT = "/kaggle/working/yolo_final"

    if os.path.exists(SPLIT):
        shutil.rmtree(SPLIT)

    for s in ["train", "val", "test"]:
        os.makedirs(f"{SPLIT}/images/{s}")
        os.makedirs(f"{SPLIT}/labels/{s}")

    def put(src_split, dst_split, fname):
        stem = os.path.splitext(fname)[0]

        for kind, name in (
            ("images", fname),
            ("labels", stem + ".txt")
        ):
            src = f"{OUT}/{kind}/{src_split}/{name}"
            dst = f"{SPLIT}/{kind}/{dst_split}/{name}"

            try:
                os.link(src, dst)
            except OSError:
                shutil.copy(src, dst)

    photos = collections.defaultdict(list)

    for f in sorted(os.listdir(f"{OUT}/images/train")):
        photos[source_key(f)].append(f)

    # Use one image per augmentation type for each training photo
    for k, files in photos.items():
        taken = collections.Counter()

        for f in files:
            t = aug_type(f)

            if t in keep_types and taken[t] < per_type:
                put("train", "train", f)
                taken[t] += 1

    # Add official validation photos to training.
    # Keep validation copies as a placeholder for YOLO.
    for f in os.listdir(f"{OUT}/images/val"):
        put("val", "train", f)
        put("val", "val", f)

    # Keep test data separate and untouched
    for f in os.listdir(f"{OUT}/images/test"):
        put("test", "test", f)

    yaml_path = "/kaggle/working/data_final.yaml"

    with open(yaml_path, "w") as f:
        f.write(
            f"path: {SPLIT}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            "  0: Aircraft\n"
            "  1: Building\n"
            "  2: Military Truck\n"
            "  3: Tank\n"
        )

    print(
        "Final training images:",
        len(os.listdir(f"{SPLIT}/images/train")),
        "| Test images:",
        len(os.listdir(f"{SPLIT}/images/test"))
    )

    print("Class mapping:")
    print("0 = Aircraft")
    print("1 = Building")
    print("2 = Military Truck")
    print("3 = Tank")

    return yaml_path


yaml_path = build_final_split(
    FINAL["keep"],
    FINAL["per_type"]
)

args = dict(
    data=yaml_path,
    epochs=FINAL["epochs"],
    patience=0,
    imgsz=FINAL["imgsz"],
    batch=FINAL["batch"],
    device=0,
    workers=2,
    project="/kaggle/working/runs",
    name="final",
    exist_ok=True,
    seed=0,
    flipud=0.5,
    cos_lr=True,
    plots=False
)

# Apply Experiment 5's training settings
args.update(FINAL["extra"])

# Train the final model
YOLO(FINAL["model"]).train(**args)

# Load the final epoch's weights
final = YOLO(
    "/kaggle/working/runs/final/weights/last.pt"
)

# Evaluate on the untouched test set
r = final.val(
    data=yaml_path,
    split="test",
    imgsz=FINAL["imgsz"],
    device=0,
    plots=False,
    verbose=False
)

print("\n===== FINAL MODEL ON THE UNTOUCHED TEST SET =====")
print(f"Precision: {r.box.mp:.3f}")
print(f"Recall:    {r.box.mr:.3f}")
print(f"mAP50:     {r.box.map50:.3f}")
print(f"mAP50-95:  {r.box.map:.3f}")

for i, c in enumerate(r.box.ap_class_index):
    print(
        f"  {final.names[int(c)]}: "
        f"AP50 = {r.box.ap50[i]:.3f}"
    )

# Save model for download
shutil.copy(
    "/kaggle/working/runs/final/weights/last.pt",
    "/kaggle/working/final_model.pt"
)

print("\nSaved /kaggle/working/final_model.pt")


# ========================================================================
# Detector training – source cell 21
# ========================================================================
# ===== ALL-IN-ONE: PREP + TOOLKIT + e7 =====
import os, re, json, time, random, shutil, subprocess, sys, collections
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
import torch
assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU T4 x2 in session options, then re-run"

# ---------- 1. copy data into YOLO layout ----------
DATA_ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "Training Data" in dirs:
        DATA_ROOT = root
        break
assert DATA_ROOT, "Dataset not found: click Add Input and add uas-aerial"

OUT = "/kaggle/working/yolo_data"
if os.path.exists(OUT):
    shutil.rmtree(OUT)
splits = {"train": "Training Data", "val": "Validation Data", "test": "Testing Data"}
for key, folder in splits.items():
    os.makedirs(f"{OUT}/images/{key}")
    os.makedirs(f"{OUT}/labels/{key}")
    for f in os.listdir(f"{DATA_ROOT}/{folder}/Images"):
        shutil.copy(f"{DATA_ROOT}/{folder}/Images/{f}", f"{OUT}/images/{key}/{f}")
    for f in os.listdir(f"{DATA_ROOT}/{folder}/Labels"):
        shutil.copy(f"{DATA_ROOT}/{folder}/Labels/{f}", f"{OUT}/labels/{key}/{f}")

NAMES = {0: "Aircraft", 1: "Building", 2: "Military Truck", 3: "Tank"}

# ---------- 2. clean labels (train and val only; test is left untouched) ----------
def clean_label_file(path):
    rows, changed = [], False
    for line in open(path):
        p = line.split()
        if not p:
            continue
        try:
            c = int(float(p[0])); v = [float(x) for x in p[1:]]
        except ValueError:
            changed = True; continue
        if len(v) == 4:
            x1, y1, x2, y2 = v[0]-v[2]/2, v[1]-v[3]/2, v[0]+v[2]/2, v[1]+v[3]/2
        elif len(v) >= 6 and len(v) % 2 == 0:          # polygon row -> bounding box
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            changed = True
        else:
            changed = True; continue
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            changed = True
        w, h = cx2 - cx1, cy2 - cy1
        if w < 0.002 or h < 0.002:
            changed = True; continue
        rows.append(f"{c} {(cx1+cx2)/2:.6f} {(cy1+cy2)/2:.6f} {w:.6f} {h:.6f}")
    return rows, changed

for split in ["train", "val"]:
    d = f"{OUT}/labels/{split}"
    fixed = 0
    for f in os.listdir(d):
        rows, changed = clean_label_file(f"{d}/{f}")
        if changed:
            fixed += 1
            with open(f"{d}/{f}", "w") as fh:
                fh.write("\n".join(rows) + ("\n" if rows else ""))
    print(f"{split}: fixed {fixed} of {len(os.listdir(d))} label files")

# ---------- 3. helpers + split builder ----------
def source_key(fname):
    m = re.match(r'^(.*?\.rf\.[0-9a-f]+)', fname)
    return m.group(1) if m else os.path.splitext(fname)[0]

def aug_type(fname):
    rest = os.path.splitext(fname)[0][len(source_key(fname)):]
    return re.sub(r'_\d+$', '', rest).lstrip('_') or "original"

ALL_TYPES = ["flip", "brightness", "haze", "shadow", "motionblur", "zoom_in", "zoom_out", "rotation"]
HELD_PATH = "/kaggle/working/held_photos.json"

def build_split(keep_types, per_type, val_fraction=0.20):
    SPLIT = "/kaggle/working/yolo_split"
    photos = collections.defaultdict(list)
    for f in sorted(os.listdir(f"{OUT}/images/train")):
        photos[source_key(f)].append(f)

    if os.path.exists(HELD_PATH):
        held = set(json.load(open(HELD_PATH)))
    else:
        dom = {}
        for k, files in photos.items():
            cnt = collections.Counter()
            stem = os.path.splitext(files[0])[0]
            for line in open(f"{OUT}/labels/train/{stem}.txt"):
                p = line.split()
                if len(p) >= 5:
                    cnt[int(float(p[0]))] += 1
            dom[k] = cnt.most_common(1)[0][0] if cnt else -1
        rng, held = random.Random(42), set()
        for c in sorted(set(dom.values())):
            keys = sorted(k for k, v in dom.items() if v == c)
            rng.shuffle(keys)
            held.update(keys[:max(1, round(val_fraction * len(keys)))])
        json.dump(sorted(held), open(HELD_PATH, "w"))

    if os.path.exists(SPLIT):
        shutil.rmtree(SPLIT)
    for s in ["train", "val", "test"]:
        os.makedirs(f"{SPLIT}/images/{s}"); os.makedirs(f"{SPLIT}/labels/{s}")

    def put(src_split, dst_split, fname):
        stem = os.path.splitext(fname)[0]
        for kind, name in (("images", fname), ("labels", stem + ".txt")):
            src, dst = f"{OUT}/{kind}/{src_split}/{name}", f"{SPLIT}/{kind}/{dst_split}/{name}"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy(src, dst)

    for k, files in photos.items():
        if k in held:
            continue
        taken = collections.Counter()
        for f in files:
            t = aug_type(f)
            if t in keep_types and taken[t] < per_type:
                put("train", "train", f); taken[t] += 1
    for f in os.listdir(f"{OUT}/images/val"):
        put("val", "val", f)
    for k in held:
        flips = [f for f in photos[k] if aug_type(f) == "flip"]
        put("train", "val", (flips or photos[k])[0])
    for f in os.listdir(f"{OUT}/images/test"):
        put("test", "test", f)

    yaml_path = "/kaggle/working/data_split.yaml"
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {SPLIT}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                 "names:\n  0: Aircraft\n  1: Building\n  2: Military Truck\n  3: Tank\n")
    for s in ["train", "val", "test"]:
        cnt = collections.Counter()
        for f in os.listdir(f"{SPLIT}/labels/{s}"):
            for line in open(f"{SPLIT}/labels/{s}/{f}"):
                p = line.split()
                if len(p) >= 5:
                    cnt[int(float(p[0]))] += 1
        print(f"  {s}: {len(os.listdir(f'{SPLIT}/images/{s}'))} images | " +
              ", ".join(f"{NAMES[c]} {cnt[c]}" for c in NAMES))
    return yaml_path

# ---------- 4. results log (earlier runs re-added so the summary table is complete) ----------
RESULTS_PATH = "/kaggle/working/results.json"
results_log = {}
_old = {  # name: (val mAP50, test P, R, mAP50, mAP50-95, (Aircraft, Building, Truck, Tank AP50), minutes)
    "exp1_ref": (0.291, 0.640, 0.360, 0.433, 0.226, (0.628, 0.633, 0.011, 0.458), 26),
    "e2_s640": (0.230, 0.732, 0.471, 0.460, 0.262, (0.679, 0.591, 0.050, 0.522), 8.5),
    "e3_s1024": (0.264, 0.739, 0.438, 0.462, 0.254, (0.727, 0.626, 0.050, 0.443), 19.5),
    "e4_s640_freeze": (0.261, 0.547, 0.425, 0.434, 0.229, (0.732, 0.513, 0.059, 0.434), 3.6),
    "e5_s640_aug": (0.292, 0.549, 0.477, 0.515, 0.278, (0.678, 0.619, 0.051, 0.711), 10.4),
    "e6_m640": (0.281, 0.814, 0.518, 0.530, 0.273, (0.736, 0.670, 0.047, 0.667), 23.8),
}
for n, (v, p, r, m50, m95, ap, mins) in _old.items():
    results_log[n] = {"minutes": mins, "val": {"map50": v},
                      "test": {"P": p, "R": r, "map50": m50, "map5095": m95,
                               "ap50": {NAMES[i]: ap[i] for i in range(4)}}}

def run_experiment(cfg):
    from ultralytics import YOLO
    t0 = time.time()
    print("\n" + "=" * 70); print("EXPERIMENT", cfg["name"]); print(cfg); print("=" * 70)
    yaml_path = build_split(cfg["keep"], cfg["per_type"])
    args = dict(data=yaml_path, epochs=cfg["epochs"], patience=30, imgsz=cfg["imgsz"],
                batch=cfg["batch"], device=0, workers=2, project="/kaggle/working/runs",
                name=cfg["name"], exist_ok=True, seed=0, flipud=0.5, cos_lr=True, plots=False)
    args.update(cfg.get("extra", {}))
    YOLO(cfg["model"]).train(**args)
    best_path = f"/kaggle/working/runs/{cfg['name']}/weights/best.pt"
    best = YOLO(best_path)
    out = {"minutes": round((time.time() - t0) / 60, 1)}
    for split in ["val", "test"]:
        r = best.val(data=yaml_path, split=split, imgsz=cfg["imgsz"], device=0, plots=False, verbose=False)
        out[split] = {"P": float(r.box.mp), "R": float(r.box.mr), "map50": float(r.box.map50),
                      "map5095": float(r.box.map),
                      "ap50": {best.names[int(c)]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}}
    shutil.copy(best_path, f"/kaggle/working/{cfg['name']}_best.pt")
    results_log[cfg["name"]] = out
    json.dump(results_log, open(RESULTS_PATH, "w"), indent=1)
    return out

def show_summary():
    head = f"{'experiment':15s}|{'VAL mAP50':>10s} |{'TEST P':>7s}{'R':>6s}{'mAP50':>7s}{'mAP5095':>8s} |"
    head += "".join(f"{NAMES[c][:8]:>9s}" for c in NAMES) + " | min"
    print(head)
    for name, o in results_log.items():
        t = o["test"]
        row = f"{name:15s}|{o['val']['map50']:10.3f} |{t['P']:7.3f}{t['R']:6.3f}{t['map50']:7.3f}{t['map5095']:8.3f} |"
        row += "".join(f"{t['ap50'].get(NAMES[c], 0):9.3f}" for c in NAMES) + f" | {o['minutes']}"
        print(row)

# ---------- 5. run e7 ----------
AUG = dict(mixup=0.15, scale=0.7, degrees=10, translate=0.2, hsv_s=0.8, hsv_v=0.5)
run_experiment(dict(name="e7_m640_aug", model="yolov8m.pt", imgsz=640, batch=16, epochs=100,
                    keep=ALL_TYPES, per_type=1, extra=AUG))
show_summary()


# ========================================================================
# Detector training – source cell 22
# ========================================================================
# ===== CELL A: PHOTO-LEVEL, CLASS-STRATIFIED 5-FOLD CROSS-VALIDATION =====
import os, re, json, time, shutil, subprocess, sys, collections
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO
assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU T4 x2, then re-run"

FOLDS, EPOCHS, MODEL, DUP = 5, 50, "yolov8s.pt", 8
AUG = dict(mixup=0.15, scale=0.7, degrees=10, translate=0.2, hsv_s=0.8, hsv_v=0.5)
NAMES = {0: "Aircraft", 1: "Building", 2: "Military Truck", 3: "Tank"}
ALL_TYPES = ["flip", "brightness", "haze", "shadow", "motionblur", "zoom_in", "zoom_out", "rotation"]

# ---- 1. copy all data into one folder tree (skipped if already there) ----
DATA_ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "Training Data" in dirs:
        DATA_ROOT = root
        break
assert DATA_ROOT, "Dataset not found: click Add Input and add uas-aerial"
OUT = "/kaggle/working/yolo_data"
if not os.path.exists(f"{OUT}/images/test"):
    splits = {"train": "Training Data", "val": "Validation Data", "test": "Testing Data"}
    for key, folder in splits.items():
        os.makedirs(f"{OUT}/images/{key}"); os.makedirs(f"{OUT}/labels/{key}")
        for f in os.listdir(f"{DATA_ROOT}/{folder}/Images"):
            shutil.copy(f"{DATA_ROOT}/{folder}/Images/{f}", f"{OUT}/images/{key}/{f}")
        for f in os.listdir(f"{DATA_ROOT}/{folder}/Labels"):
            shutil.copy(f"{DATA_ROOT}/{folder}/Labels/{f}", f"{OUT}/labels/{key}/{f}")

# ---- 2. clean labels: clip boxes to the image, turn polygon rows into boxes ----
def clean_label_file(path):
    rows, changed = [], False
    for line in open(path):
        p = line.split()
        if not p:
            continue
        try:
            c = int(float(p[0])); v = [float(x) for x in p[1:]]
        except ValueError:
            changed = True; continue
        if len(v) == 4:
            x1, y1, x2, y2 = v[0]-v[2]/2, v[1]-v[3]/2, v[0]+v[2]/2, v[1]+v[3]/2
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys); changed = True
        else:
            changed = True; continue
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            changed = True
        w, h = cx2 - cx1, cy2 - cy1
        if w < 0.002 or h < 0.002:
            changed = True; continue
        rows.append(f"{c} {(cx1+cx2)/2:.6f} {(cy1+cy2)/2:.6f} {w:.6f} {h:.6f}")
    return rows, changed

nfix = 0
for split in ["train", "val", "test"]:
    d = f"{OUT}/labels/{split}"
    for f in os.listdir(d):
        rows, changed = clean_label_file(f"{d}/{f}")
        if changed:
            nfix += 1
            open(f"{d}/{f}", "w").write("\n".join(rows) + ("\n" if rows else ""))
print("label files fixed:", nfix)

# ---- 3. build the list of original photos ("scenes") ----
def source_key(fname):
    m = re.match(r'^(.*?\.rf\.[0-9a-f]+)', fname)
    return m.group(1) if m else os.path.splitext(fname)[0]

def aug_type(fname):
    rest = os.path.splitext(fname)[0][len(source_key(fname)):]
    return re.sub(r'_\d+$', '', rest).lstrip('_') or "original"

scenes = []
photos = collections.defaultdict(list)
for f in sorted(os.listdir(f"{OUT}/images/train")):
    photos[source_key(f)].append(f)
for k, files in photos.items():
    taken, tr = collections.Counter(), []
    for f in files:
        t = aug_type(f)
        if t in ALL_TYPES and taken[t] < 1:
            tr.append(("train", f, 1)); taken[t] += 1
    flips = [f for f in files if aug_type(f) == "flip"]
    scenes.append(dict(id=k, cls=k.split("-")[0].lower(), train=tr, eval=("train", (flips or files)[0])))
for split in ["val", "test"]:
    for f in sorted(os.listdir(f"{OUT}/images/{split}")):
        scenes.append(dict(id=f"{split}_{source_key(f)}", cls=f.split("-")[0].lower(),
                           train=[(split, f, DUP)], eval=(split, f)))
print("original photos:", len(scenes), "| per class:", dict(collections.Counter(s["cls"] for s in scenes)))

# ---- 4. dataset builder ----
def build_dataset(D, train_scenes, val_scenes):
    if os.path.exists(D):
        shutil.rmtree(D)
    for s in ["train", "val"]:
        os.makedirs(f"{D}/images/{s}"); os.makedirs(f"{D}/labels/{s}")
    def put(split, fname, dst_split, copies=1):
        stem, ext = os.path.splitext(fname)
        for c in range(copies):
            nm = stem if copies == 1 else f"{stem}_c{c}"
            for kind, sname, dname in (("images", fname, nm + ext), ("labels", stem + ".txt", nm + ".txt")):
                src, dst = f"{OUT}/{kind}/{split}/{sname}", f"{D}/{kind}/{dst_split}/{dname}"
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy(src, dst)
    for sc in train_scenes:
        for (sp, f, cp) in sc["train"]:
            put(sp, f, "train", cp)
    for sc in val_scenes:
        put(sc["eval"][0], sc["eval"][1], "val")
    yaml_path = f"{D}/data.yaml"
    open(yaml_path, "w").write(f"path: {D}\ntrain: images/train\nval: images/val\n\n"
        "names:\n  0: Aircraft\n  1: Building\n  2: Military Truck\n  3: Tank\n")
    return yaml_path

def train_args(yaml_path, name, project):
    a = dict(data=yaml_path, epochs=EPOCHS, patience=0, imgsz=640, batch=16, device=0, workers=2,
             project=project, name=name, exist_ok=True, seed=0, flipud=0.5, cos_lr=True,
             plots=False, val=False)      # val=False: no validation until the last epoch, no selection on held-out photos
    a.update(AUG)
    return a

# ---- 5. cross-validation ----
CV_PATH = "/kaggle/working/cv_results.json"
cv = json.load(open(CV_PATH)) if os.path.exists(CV_PATH) else {}
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
y = [s["cls"] for s in scenes]
for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
    if f"fold{k}" in cv:
        print(f"fold {k}: already done, skipping"); continue
    t0 = time.time()
    tr_sc, va_sc = [scenes[i] for i in tr_idx], [scenes[i] for i in va_idx]
    print(f"\n=== FOLD {k}/{FOLDS}: train {len(tr_sc)} photos, test {len(va_sc)} photos "
          f"({dict(collections.Counter(s['cls'] for s in va_sc))}) ===")
    D = f"/kaggle/working/cv/fold{k}"
    yaml_path = build_dataset(D, tr_sc, va_sc)
    YOLO(MODEL).train(**train_args(yaml_path, f"fold{k}", "/kaggle/working/cv_runs"))
    m = YOLO(f"/kaggle/working/cv_runs/fold{k}/weights/last.pt")
    r = m.val(data=yaml_path, split="val", imgsz=640, device=0, plots=False, verbose=False)
    cv[f"fold{k}"] = {"P": float(r.box.mp), "R": float(r.box.mr), "map50": float(r.box.map50),
                      "map5095": float(r.box.map),
                      "ap50": {NAMES[int(c)]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)},
                      "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(cv, open(CV_PATH, "w"), indent=1)
    shutil.rmtree(D, ignore_errors=True)
    print(f"fold {k} done in {cv[f'fold{k}']['minutes']} min | mAP50 {cv[f'fold{k}']['map50']:.3f}")

# ---- 6. summary ----
print("\n" + "=" * 78)
print(f"{FOLDS}-FOLD CROSS-VALIDATION ({MODEL}, {EPOCHS} epochs, every photo tested once)")
print("=" * 78)
print(f"{'fold':6s}{'P':>7s}{'R':>7s}{'mAP50':>8s}{'mAP50-95':>10s}" + "".join(f"{NAMES[c][:8]:>10s}" for c in NAMES))
for k in range(1, FOLDS + 1):
    o = cv.get(f"fold{k}")
    if o:
        print(f"{k:<6d}{o['P']:7.3f}{o['R']:7.3f}{o['map50']:8.3f}{o['map5095']:10.3f}" +
              "".join(f"{o['ap50'].get(NAMES[c], float('nan')):10.3f}" for c in NAMES))
vals = [cv[f"fold{k}"] for k in range(1, FOLDS + 1) if f"fold{k}" in cv]
def ms(key):
    a = np.array([v[key] for v in vals]); return a.mean(), a.std()
print("\nMEAN +- STD ACROSS FOLDS")
for key, label in (("P", "Precision"), ("R", "Recall"), ("map50", "mAP50"), ("map5095", "mAP50-95")):
    mu, sd = ms(key); print(f"  {label:10s}{mu:.3f} +- {sd:.3f}")
for c in NAMES:
    a = np.array([v["ap50"][NAMES[c]] for v in vals if NAMES[c] in v["ap50"]])
    print(f"  AP50 {NAMES[c]:15s}{a.mean():.3f} +- {a.std():.3f}")


# ========================================================================
# Detector training – source cell 23
# ========================================================================
# ===== CELL B: FINAL MODEL ON ALL 120 PHOTOS =====
D = "/kaggle/working/final_data"
yaml_path = build_dataset(D, scenes, scenes[:16])      # val copy is only a placeholder for YOLO, never scored
YOLO(MODEL).train(**train_args(yaml_path, "final_all", "/kaggle/working/runs"))
shutil.copy("/kaggle/working/runs/final_all/weights/last.pt", "/kaggle/working/final_model_all.pt")
print("Saved /kaggle/working/final_model_all.pt  <- download this")
