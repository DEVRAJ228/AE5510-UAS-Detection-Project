# Source-derived dataset preparation and audit cells from uas-yolo.
# Kaggle paths and dataset inputs are expected; see README.md.


# ========================================================================
# Dataset preparation – source cell 4
# ========================================================================
import os
DATA_ROOT = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "Training Data" in dirs:
        DATA_ROOT = root
        break
print("DATA_ROOT =", DATA_ROOT)


# ========================================================================
# Dataset preparation – source cell 5
# ========================================================================
splits = {"train": "Training Data", "val": "Validation Data", "test": "Testing Data"}
for key, folder in splits.items():
    imgs = os.listdir(f"{DATA_ROOT}/{folder}/Images")
    lbls = os.listdir(f"{DATA_ROOT}/{folder}/Labels")
    print(key, "| images:", len(imgs), "| labels:", len(lbls))

print(sorted(os.listdir(f"{DATA_ROOT}/Training Data/Images"))[:8])


# ========================================================================
# Dataset preparation – source cell 6
# ========================================================================
import shutil
OUT = "/kaggle/working/yolo_data"
if os.path.exists(OUT):
    shutil.rmtree(OUT)

for key, folder in splits.items():
    os.makedirs(f"{OUT}/images/{key}")
    os.makedirs(f"{OUT}/labels/{key}")
    for f in os.listdir(f"{DATA_ROOT}/{folder}/Images"):
        shutil.copy(f"{DATA_ROOT}/{folder}/Images/{f}", f"{OUT}/images/{key}/{f}")
    for f in os.listdir(f"{DATA_ROOT}/{folder}/Labels"):
        shutil.copy(f"{DATA_ROOT}/{folder}/Labels/{f}", f"{OUT}/labels/{key}/{f}")

yaml_text = """path: /kaggle/working/yolo_data
train: images/train
val: images/val
test: images/test

names:
  0: Aircraft
  1: Building
  2: Military Truck
  3: Tank
"""
with open("/kaggle/working/data.yaml", "w") as f:
    f.write(yaml_text)
print("Done. data.yaml written.")


# ========================================================================
# Dataset preparation – source cell 7
# ========================================================================
import cv2, random, collections
import matplotlib.pyplot as plt

names = {0: "Aircraft", 1: "Building", 2: "Military Truck", 3: "Tank"}
lbl_dir = f"{OUT}/labels/train"
counts = collections.Counter()
examples = {c: [] for c in names}

for f in os.listdir(lbl_dir):
    for line in open(f"{lbl_dir}/{f}"):
        p = line.split()
        if len(p) >= 5:
            c = int(float(p[0]))
            counts[c] += 1
            examples[c].append(f)

for c in names:
    print(f"class {c} = {names[c]}: {counts[c]} objects")

def find_img(stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = f"{OUT}/images/train/{stem}{ext}"
        if os.path.exists(p):
            return p

fig, axs = plt.subplots(1, 4, figsize=(22, 6))
for ax, c in zip(axs, names):
    f = random.choice(examples[c])
    img = cv2.cvtColor(cv2.imread(find_img(os.path.splitext(f)[0])), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for line in open(f"{lbl_dir}/{f}"):
        p = line.split()
        if int(float(p[0])) == c:
            xc, yc, w, h = [float(v) for v in p[1:5]]
            cv2.rectangle(img, (int((xc-w/2)*W), int((yc-h/2)*H)),
                          (int((xc+w/2)*W), int((yc+h/2)*H)), (255, 0, 0), 3)
    ax.imshow(img)
    ax.set_title(f"class {c} = {names[c]}")
    ax.axis("off")
plt.show()


# ========================================================================
# Dataset preparation – source cell 8
# ========================================================================
import re, collections

def source_key(fname):
    m = re.match(r'^(.*?\.rf\.[0-9a-f]+)', fname)
    return m.group(1) if m else os.path.splitext(fname)[0]

def keys_of(split):
    return [source_key(f) for f in os.listdir(f"{OUT}/images/{split}")]

tr, va, te = keys_of("train"), keys_of("val"), keys_of("test")
tr_set = set(tr)
cnt = collections.Counter(tr)

print("train images:", len(tr), "| unique original photos:", len(tr_set))
print("variants per photo: min", min(cnt.values()), "max", max(cnt.values()))
print("val photos also in train:", len(set(va) & tr_set), "of", len(set(va)))
print("test photos also in train:", len(set(te) & tr_set), "of", len(set(te)))
print("val/test overlap:", len(set(va) & set(te)))

kinds = collections.Counter()
for f in os.listdir(f"{OUT}/images/train"):
    base = os.path.splitext(f)[0]
    rest = base[len(source_key(f)):]
    kinds[re.sub(r'_\d+$', '', rest) or "original"] += 1
print("augmentation types:", kinds.most_common())

print("val examples:", sorted(os.listdir(f"{OUT}/images/val"))[:4])
print("test examples:", sorted(os.listdir(f"{OUT}/images/test"))[:4])


# ========================================================================
# Dataset preparation – source cell 9
# ========================================================================
# ===== CELL 8: ALL-IN-ONE PREP =====
import os, shutil, subprocess, sys, collections
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

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
print("Dataset:", DATA_ROOT)
for key in splits:
    print(key, "images:", len(os.listdir(f"{OUT}/images/{key}")),
          "| labels:", len(os.listdir(f"{OUT}/labels/{key}")))
print("Class IDs:", NAMES)
print("PREP DONE")


# ========================================================================
# Dataset preparation – source cell 10
# ========================================================================
# ===== CELL 9: PHOTO-LEVEL SPLIT =====
import re, random, collections, os, shutil
random.seed(42)

VAL_FRACTION = 0.20
KEEP_TYPES = None   # None = keep all augmentations.
                    # Later try e.g. ["flip","brightness","rotation","zoom_in","zoom_out"]

def source_key(fname):
    m = re.match(r'^(.*?\.rf\.[0-9a-f]+)', fname)
    return m.group(1) if m else os.path.splitext(fname)[0]

def aug_type(fname):
    base = os.path.splitext(fname)[0]
    rest = re.sub(r'_\d+$', '', base[len(source_key(fname)):]).lstrip('_')
    return rest or "original"

def label_path(split_dir, fname):
    return f"{split_dir}/labels/{os.path.splitext(fname)[0]}.txt"

def link(src, dst):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy(src, dst)

# group train images by original photo
photos = collections.defaultdict(list)
for f in os.listdir(f"{OUT}/images/train"):
    photos[source_key(f)].append(f)

# dominant class of each photo (for a stratified split)
dom = {}
for k, files in photos.items():
    c = collections.Counter()
    for line in open(label_path(OUT, files[0]).replace("/labels/", "/labels/train/")):
        if line.strip():
            c[int(float(line.split()[0]))] += 1
    dom[k] = c.most_common(1)[0][0] if c else -1

by_class = collections.defaultdict(list)
for k, c in dom.items():
    by_class[c].append(k)

held = set()
for c, keys in by_class.items():
    random.shuffle(keys)
    n = max(1, round(VAL_FRACTION * len(keys)))
    held.update(keys[:n])

SPLIT = "/kaggle/working/yolo_split"
if os.path.exists(SPLIT):
    shutil.rmtree(SPLIT)
for s in ["train", "val", "test"]:
    os.makedirs(f"{SPLIT}/images/{s}")
    os.makedirs(f"{SPLIT}/labels/{s}")

def put(split_src, split_dst, fname):
    link(f"{OUT}/images/{split_src}/{fname}", f"{SPLIT}/images/{split_dst}/{fname}")
    lbl = os.path.splitext(fname)[0] + ".txt"
    link(f"{OUT}/labels/{split_src}/{lbl}", f"{SPLIT}/labels/{split_dst}/{lbl}")

# train: all (or chosen) variants of non-held photos
for k, files in photos.items():
    if k in held:
        continue
    for f in files:
        if KEEP_TYPES is None or aug_type(f) in KEEP_TYPES:
            put("train", "train", f)

# val: official 16 clean val images + the flip variant of each held-out photo
for f in os.listdir(f"{OUT}/images/val"):
    put("val", "val", f)
for k in held:
    flips = [f for f in photos[k] if aug_type(f) == "flip"]
    put("train", "val", (flips or photos[k])[0])

# test: official test, untouched
for f in os.listdir(f"{OUT}/images/test"):
    put("test", "test", f)

yaml_text = """path: /kaggle/working/yolo_split
train: images/train
val: images/val
test: images/test

names:
  0: Aircraft
  1: Building
  2: Military Truck
  3: Tank
"""
open("/kaggle/working/data_split.yaml", "w").write(yaml_text)

# report
def count_objs(s):
    c = collections.Counter()
    for f in os.listdir(f"{SPLIT}/labels/{s}"):
        for line in open(f"{SPLIT}/labels/{s}/{f}"):
            if line.strip():
                c[int(float(line.split()[0]))] += 1
    return c

print("Photos total:", len(photos), "| held out for val:", len(held))
print("Held-out photos per class:", {NAMES[c]: sum(dom[k] == c for k in held) for c in NAMES})
for s in ["train", "val", "test"]:
    oc = count_objs(s)
    print(f"{s}: {len(os.listdir(f'{SPLIT}/images/{s}'))} images | objects:",
          {NAMES[c]: oc[c] for c in NAMES})
