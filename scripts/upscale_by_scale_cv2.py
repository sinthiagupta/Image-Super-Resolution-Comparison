# scripts/upscale_sample_5.py
"""
Upscale SAMPLE 5 images only using Nearest / Bilinear / Bicubic.
Saves individual method outputs and a combined comparison image per sample:
  outputs/sample/image_001_compare.png  (LR | Nearest | Bilinear | Bicubic | HR)
Config:
  SPLIT = 'train'|'val'|'test'
  MODE = 'scale' or 'match_hr'
    - 'scale': multiply LR w/h by SCALE_FACTOR (like your example)
    - 'match_hr': resize LR to exact HR size (if HR sizes vary)
"""

import os
import math
import random
from PIL import Image
import cv2
import numpy as np

# ---------------- CONFIG ----------------
SPLIT = "train"           # 'train' / 'val' / 'test'
MODE = "match_hr"         # 'scale' or 'match_hr'
SCALE_FACTOR = 4.0        # used only if MODE == 'scale'
NUM_SAMPLES = 5           # number of images to process
SELECT_FIRST = True       # if True: take first NUM_SAMPLES; else random.sample
LR_DIR = os.path.join("data", "processed", SPLIT, "LR")
HR_DIR = os.path.join("data", "processed", SPLIT, "HR")

OUT_BASE = os.path.join("outputs", "sample")
OUT_NEAREST = os.path.join(OUT_BASE, "nearest")
OUT_BILINEAR = os.path.join(OUT_BASE, "bilinear")
OUT_BICUBIC = os.path.join(OUT_BASE, "bicubic")
OUT_COMPARE = os.path.join(OUT_BASE, "compare")

for d in (OUT_NEAREST, OUT_BILINEAR, OUT_BICUBIC, OUT_COMPARE):
    os.makedirs(d, exist_ok=True)
# ----------------------------------------

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])

def find_matching_hr(lr_name, hr_list):
    # try exact match
    if lr_name in hr_list:
        return lr_name
    base, _ = os.path.splitext(lr_name)
    # strip common suffixes
    for suf in ("_lr","-lr","_LR","-LR","_small","-small"):
        if base.endswith(suf):
            base = base[:-len(suf)]; break
    for ext in (".png",".jpg",".jpeg"):
        cand = base + ext
        if cand in hr_list: return cand
    for h in hr_list:
        if h.startswith(base): return h
    return None

def upscale_by_scale_cv2(lr_bgr, new_w, new_h):
    near = cv2.resize(lr_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    bil = cv2.resize(lr_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bic = cv2.resize(lr_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return near, bil, bic

def make_compare_strip(lr_pil, near_pil, bil_pil, bic_pil, hr_pil):
    # All inputs are PIL.Image in RGB and same size
    w, h = lr_pil.size
    pad = 5
    total_w = (w * 5) + (pad * 6)
    out = Image.new("RGB", (total_w, h + pad*2), (255,255,255))
    x = pad
    for im in (lr_pil, near_pil, bil_pil, bic_pil, hr_pil):
        out.paste(im, (x, pad))
        x += w + pad
    return out

# --- main ---
lr_files = list_images(LR_DIR)
hr_files = list_images(HR_DIR)
if len(lr_files) == 0 or len(hr_files) == 0:
    raise SystemExit("LR or HR folder empty. Check data/processed/<split>/*")

# select files
if SELECT_FIRST:
    chosen = lr_files[:NUM_SAMPLES]
else:
    random.seed(1234)
    chosen = random.sample(lr_files, NUM_SAMPLES)

print("Selected images:", chosen)

for idx, lr_name in enumerate(chosen, start=1):
    match_hr = find_matching_hr(lr_name, hr_files)
    if match_hr is None:
        print("Skipping (no HR match):", lr_name); continue

    lr_path = os.path.join(LR_DIR, lr_name)
    hr_path = os.path.join(HR_DIR, match_hr)

    # read LR and HR
    lr_pil = Image.open(lr_path).convert("RGB")
    hr_pil = Image.open(hr_path).convert("RGB")

    if MODE == "scale":
        w, h = lr_pil.size
        new_w = max(1, int(round(w * SCALE_FACTOR)))
        new_h = max(1, int(round(h * SCALE_FACTOR)))
    else:  # match_hr
        new_w, new_h = hr_pil.size

    # use OpenCV for fast accurate interpolation (BGR)
    lr_bgr = cv2.cvtColor(np.array(lr_pil), cv2.COLOR_RGB2BGR)
    near_bgr, bil_bgr, bic_bgr = upscale_by_scale_cv2(lr_bgr, new_w, new_h)

    # convert back to PIL RGB
    near_pil = Image.fromarray(cv2.cvtColor(near_bgr, cv2.COLOR_BGR2RGB))
    bil_pil  = Image.fromarray(cv2.cvtColor(bil_bgr, cv2.COLOR_BGR2RGB))
    bic_pil  = Image.fromarray(cv2.cvtColor(bic_bgr, cv2.COLOR_BGR2RGB))

    # save individual outputs
    base = f"sample_{idx:02d}"
    near_pil.save(os.path.join(OUT_NEAREST, f"{base}_nearest.png"))
    bil_pil.save(os.path.join(OUT_BILINEAR, f"{base}_bilinear.png"))
    bic_pil.save(os.path.join(OUT_BICUBIC, f"{base}_bicubic.png"))
    # also save LR and HR copies under compare folder for convenience
    lr_pil.save(os.path.join(OUT_COMPARE, f"{base}_lr.png"))
    hr_pil.save(os.path.join(OUT_COMPARE, f"{base}_hr.png"))

    # create combined comparison strip and save
    # ensure all images have same size (resize LR display to new_w,new_h for montage)
    lr_display = lr_pil.resize((new_w, new_h), Image.NEAREST)
    compare_img = make_compare_strip(lr_display, near_pil, bil_pil, bic_pil, hr_pil.resize((new_w,new_h)))
    compare_img.save(os.path.join(OUT_COMPARE, f"{base}_compare.png"))

    print(f"Saved sample {idx}: {lr_name} -> nearest/bilinear/bicubic + compare image")

print("Done. Check outputs in:", OUT_BASE)
