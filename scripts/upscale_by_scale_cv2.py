# convert_make_lr_and_upscale_bicubic.py
# Creates LR by downsampling HR with BICUBIC, then upscales LR -> HR using
# Nearest, Bilinear and Bicubic (so Bicubic reconstruction is best).
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from PIL import Image

# === SETTINGS ===
HR_DIR = Path("data/processed/test/hr")      # source HR images (use these)
LR_OUT = Path("data/processed/test/LR_generated")  # generated LR (bicubic downsample)
OUT_BASE = Path("outputs/sample")
OUT_NEAR = OUT_BASE / "nearest"
OUT_BIL  = OUT_BASE / "bilinear"
OUT_BIC  = OUT_BASE / "bicubic"

SCALE = 4   # down/up scale factor (change to 6 if you want larger upscaling)
N = 5       # number of first HR images to process
# =================

for d in (LR_OUT, OUT_NEAR, OUT_BIL, OUT_BIC):
    d.mkdir(parents=True, exist_ok=True)

# list first N HR images (sorted)
hr_files = sorted([p for p in HR_DIR.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])[:N]
if not hr_files:
    raise SystemExit(f"No HR images found in {HR_DIR}")

print(f"Processing {len(hr_files)} images (scale={SCALE}). LR created by BICUBIC downsampling -> then upscaled by 3 methods.")

for idx, hr_path in enumerate(hr_files, start=1):
    prefix = f"sample_{idx:02d}"
    # load HR
    hr_img = Image.open(hr_path).convert("RGB")
    w,h = hr_img.size

    # 1) create LR by downsampling with BICUBIC
    lr_w, lr_h = max(1, w // SCALE), max(1, h // SCALE)
    lr_img = hr_img.resize((lr_w, lr_h), resample=Image.BICUBIC)
    lr_path = LR_OUT / hr_path.name.replace(".png","_lr.png").replace(".jpg","_lr.png")
    lr_img.save(lr_path)
    print(f"[{idx:02d}] Created LR (bicubic) -> {lr_path.name}  ({lr_w}x{lr_h})")

    # 2) upscale LR -> new_HR_size using three methods (OpenCV)
    lr_arr = np.array(lr_img)  # RGB
    bgr = cv2.cvtColor(lr_arr, cv2.COLOR_RGB2BGR)
    new_w, new_h = w, h

    near = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    bil  = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    bic  = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # convert BGR->RGB and save PNGs
    near_rgb = cv2.cvtColor(near, cv2.COLOR_BGR2RGB)
    bil_rgb  = cv2.cvtColor(bil,  cv2.COLOR_BGR2RGB)
    bic_rgb  = cv2.cvtColor(bic,  cv2.COLOR_BGR2RGB)

    Image.fromarray(near_rgb).save(OUT_NEAR / f"{prefix}_nearest.png")
    Image.fromarray(bil_rgb).save(OUT_BIL  / f"{prefix}_bilinear.png")
    Image.fromarray(bic_rgb).save(OUT_BIC  / f"{prefix}_bicubic.png")

    print(f"         Saved: {prefix}_nearest.png, {prefix}_bilinear.png, {prefix}_bicubic.png")

print("\nDone. Check folders:")
print(" - LR generated:", LR_OUT)
print(" - Nearest outputs:", OUT_NEAR)
print(" - Bilinear outputs:", OUT_BIL)
print(" - Bicubic outputs:", OUT_BIC)