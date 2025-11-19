# evaluate_all_methods_run.py
import cv2, csv, time
import numpy as np
from pathlib import Path
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# ---------- CONFIG ----------
HR_DIR = Path("data/processed/test/hr")
SRCNN_DIR = Path("outputs/images")               # expects files like 0091_pred.png
NEAR_DIR = Path("outputs/sample/nearest")
BIL_DIR  = Path("outputs/sample/bilinear")
BIC_DIR  = Path("outputs/sample/bicubic")

N = 5                     # number of first HR images to evaluate
OUT_CSV = Path("evaluation_results.csv")
# --------------------------

def list_hr_files(folder: Path, N):
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")])
    return files[:N]

def load_y(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

def compute_metrics(hr_y, pred_y):
    if hr_y.shape != pred_y.shape:
        pred_y = cv2.resize(pred_y, (hr_y.shape[1], hr_y.shape[0]), interpolation=cv2.INTER_CUBIC)
    mse = float(mean_squared_error(hr_y, pred_y))
    psnr = float(peak_signal_noise_ratio(hr_y, pred_y, data_range=255))
    ssim = float(structural_similarity(hr_y, pred_y, data_range=255))
    return mse, psnr, ssim

def find_srcnn_pred_for(hr_path):
    candidates = [
        SRCNN_DIR / f"{hr_path.stem}_pred.png",
        SRCNN_DIR / f"{hr_path.stem}_pred.jpg",
        SRCNN_DIR / f"{hr_path.stem}.pred.png",
        SRCNN_DIR / f"{hr_path.stem}.png".replace("_hr","_pred"),
        SRCNN_DIR / f"{hr_path.stem}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    # try numeric pattern
    try:
        num = int(''.join(ch for ch in hr_path.stem if ch.isdigit()))
        cand = SRCNN_DIR / f"{num:04d}_pred.png"
        if cand.exists(): return cand
    except Exception:
        pass
    return None

# --- MAIN ---
hr_files = list_hr_files(HR_DIR, N)
if not hr_files:
    raise SystemExit(f"No HR files found in {HR_DIR}. Put HR images and retry.")

methods = ["SRCNN","Nearest","Bilinear","Bicubic"]
acc = {m:[] for m in methods}
rows = []

print(f"Evaluating {len(hr_files)} HR images...\n")

for idx, hr_path in enumerate(hr_files, start=1):
    sample_i = idx
    hr_y = load_y(hr_path)
    if hr_y is None:
        print(f"Missing HR: {hr_path}; skipping")
        continue

    # SRCNN
    srcnn_p = find_srcnn_pred_for(hr_path)
    if srcnn_p and srcnn_p.exists():
        t0 = time.time()
        pred = load_y(srcnn_p)
        if pred is not None:
            mse, psnr, ssim = compute_metrics(hr_y, pred)
            acc["SRCNN"].append((mse, psnr, ssim))
            rows.append(["SRCNN", hr_path.name, mse, psnr, ssim, f"{time.time()-t0:.4f}"])
    else:
        print(f" SRCNN pred not found for {hr_path.name} (looked in {SRCNN_DIR})")

    # Nearest
    near_p = NEAR_DIR / f"sample_{sample_i:02d}_nearest.png"
    if near_p.exists():
        t0=time.time(); pred=load_y(near_p)
        mse,psnr,ssim = compute_metrics(hr_y, pred)
        acc["Nearest"].append((mse,psnr,ssim)); rows.append(["Nearest", hr_path.name, mse, psnr, ssim, f"{time.time()-t0:.4f}"])
    else:
        print(f" Nearest missing: {near_p}")

    # Bilinear
    bil_p = BIL_DIR / f"sample_{sample_i:02d}_bilinear.png"
    if bil_p.exists():
        t0=time.time(); pred=load_y(bil_p)
        mse,psnr,ssim = compute_metrics(hr_y, pred)
        acc["Bilinear"].append((mse,psnr,ssim)); rows.append(["Bilinear", hr_path.name, mse, psnr, ssim, f"{time.time()-t0:.4f}"])
    else:
        print(f" Bilinear missing: {bil_p}")

    # Bicubic
    bic_p = BIC_DIR / f"sample_{sample_i:02d}_bicubic.png"
    if bic_p.exists():
        t0=time.time(); pred=load_y(bic_p)
        mse,psnr,ssim = compute_metrics(hr_y, pred)
        acc["Bicubic"].append((mse,psnr,ssim)); rows.append(["Bicubic", hr_path.name, mse, psnr, ssim, f"{time.time()-t0:.4f}"])
    else:
        print(f" Bicubic missing: {bic_p}")

# write CSV (per-image + averages)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["method","image","mse","psnr","ssim","eval_time_s"])
    writer.writerows(rows)
    writer.writerow([])
    for m in methods:
        vals = acc[m]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            writer.writerow([m,"AVERAGE", float(arr[:,0].mean()), float(arr[:,1].mean()), float(arr[:,2].mean()), "0.0"])
        else:
            writer.writerow([m,"AVERAGE","NA","NA","NA","NA"])

# print copy-paste friendly summary
print("\n=== AVERAGE RESULTS (copy to PPT) ===")
for m in methods:
    vals = acc[m]
    if not vals:
        print(f"{m:8s} | NO DATA")
        continue
    arr = np.array(vals, dtype=np.float64)
    mse_avg = arr[:,0].mean(); psnr_avg = arr[:,1].mean(); ssim_avg = arr[:,2].mean()
    print(f"{m:8s} | Avg MSE = {mse_avg:.4f} | Avg PSNR = {psnr_avg:.3f} dB | Avg SSIM = {ssim_avg:.4f}")

print(f"\nPer-image results + averages saved to: {OUT_CSV}")