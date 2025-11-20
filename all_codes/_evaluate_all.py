import cv2
import numpy as np
import csv
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

HR_DIR = "outputs/images"
PRED_DIR = "outputs/images"
N = 5  # number of images (0091–0095)

CSV_PATH = "srcnn_results.csv"

mse_list = []
psnr_list = []
ssim_list = []

rows = []

for num in range(91, 91 + N):  # 0091 - 0095
    hr_path = f"{HR_DIR}/{num:04d}_hr.png"
    pred_path = f"{PRED_DIR}/{num:04d}_pred.png"

    hr = cv2.imread(hr_path)
    pred = cv2.imread(pred_path)

    if hr is None or pred is None:
        print(f"Missing: {num:04d} — skipping")
        continue

    # Convert to Y channel
    hr_y = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    pred_y = cv2.cvtColor(pred, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # Compute metrics
    mse = mean_squared_error(hr_y, pred_y)
    psnr = peak_signal_noise_ratio(hr_y, pred_y, data_range=255)
    ssim = structural_similarity(hr_y, pred_y, data_range=255)

    mse_list.append(mse)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    rows.append([f"{num:04d}", mse, psnr, ssim])

# Compute averages
avg_mse = float(np.mean(mse_list))
avg_psnr = float(np.mean(psnr_list))
avg_ssim = float(np.mean(ssim_list))

rows.append(["AVERAGE", avg_mse, avg_psnr, avg_ssim])

# Save CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "MSE", "PSNR", "SSIM"])
    writer.writerows(rows)

print("\nSaved results to:", CSV_PATH)
print(f"Avg MSE  = {avg_mse:.6f}")
print(f"Avg PSNR = {avg_psnr:.3f} dB")
print(f"Avg SSIM = {avg_ssim:.4f}")
