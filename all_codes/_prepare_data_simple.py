# scripts/preprocess_fixed_first80_81_90.py
"""
Deterministic preprocess:
 - read HR images from data/HR/ (sorted order)
 - create blurry LR images in data/LR/ (HR -> downscale -> upscale)
 - split by fixed indices:
     train = first 80 images (indices 0..79)
     val   = next 10 images  (indices 80..89)
     test  = next 10 images  (indices 90..99)
   (if fewer than 100 images, uses available counts)
 - copy paired HR and LR into data/processed/{train,val,test}/{HR,LR}

Run:
    python scripts/preprocess_fixed_first80_81_90.py
"""
import os
from PIL import Image
from shutil import copyfile

# --- NEW: plotting libs for summary ---
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
# ---------------------------------------

# -------------- SETTINGS --------------
SCALE = 4                    # downscale factor for LR creation
TRAIN_END = 80               # first 80 -> train (indices 0..79)
VAL_END = 90                 # 81..90 -> val (indices 80..89), test starts at 90
# --------------------------------------

ROOT = os.path.dirname(os.path.dirname(__file__))
HR_FOLDER = os.path.join(ROOT, "data", "HR")
LR_FOLDER = os.path.join(ROOT, "data", "LR")              # intermediate LR images
PROC_FOLDER = os.path.join(ROOT, "data", "processed")

# helper to list image files (sorted)
def list_images(folder):
    if not os.path.isdir(folder):
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

# create required folders
def make_dirs():
    os.makedirs(LR_FOLDER, exist_ok=True)
    for split in ("train","val","test"):
        os.makedirs(os.path.join(PROC_FOLDER, split, "HR"), exist_ok=True)
        os.makedirs(os.path.join(PROC_FOLDER, split, "LR"), exist_ok=True)

# create blurry LR files for each HR (same base name + _lr.png)
def create_blurry_lr_files(hr_files):
    if not hr_files:
        raise SystemExit(f"No HR images found in {HR_FOLDER}. Put your HR images there and run again.")
    print(f"Creating blurry LR images for {len(hr_files)} HR files (saved to {LR_FOLDER}) ...")
    for fname in hr_files:
        src = os.path.join(HR_FOLDER, fname)
        base, _ext = os.path.splitext(fname)
        dst = os.path.join(LR_FOLDER, base + "_lr.png")
        try:
            img = Image.open(src).convert("RGB")
            w, h = img.size
            small = img.resize((max(1, w//SCALE), max(1, h//SCALE)), Image.BICUBIC)
            lr = small.resize((w, h), Image.BICUBIC)
            lr.save(dst, format="PNG")
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

# deterministic fixed split and copy pairs
def fixed_split_copy(hr_files):
    # ensure LR files exist (we only use HR list order)
    lr_files = [os.path.splitext(f)[0] + "_lr.png" for f in hr_files]
    n = min(len(hr_files), len(lr_files))
    if n == 0:
        raise SystemExit("No files to process after checking HR/LR lists.")

    print(f"\nUsing first {n} files for splitting (master = HR list).")

    for idx in range(n):
        if idx < TRAIN_END:
            split = "train"
        elif idx < VAL_END:
            split = "val"
        else:
            split = "test"

        hr_name = hr_files[idx]
        lr_name = lr_files[idx]

        src_hr = os.path.join(HR_FOLDER, hr_name)
        src_lr = os.path.join(LR_FOLDER, lr_name)
        dst_hr = os.path.join(PROC_FOLDER, split, "HR", hr_name)
        dst_lr = os.path.join(PROC_FOLDER, split, "LR", lr_name)

        try:
            copyfile(src_hr, dst_hr)
            copyfile(src_lr, dst_lr)
            # print one line per 10 copies to reduce noise (optional)
            if (idx + 1) % 10 == 0 or idx < 5:
                print(f"[{split}] idx {idx+1}: copied HR->{hr_name}  LR->{lr_name}")
        except Exception as e:
            print(f"  Failed to copy index {idx+1} ({hr_name}, {lr_name}): {e}")

    # final counts
    def count(folder):
        if not os.path.isdir(folder):
            return 0
        exts = (".png",".jpg",".jpeg",".bmp",".tiff")
        return len([f for f in os.listdir(folder) if f.lower().endswith(exts)])

    print("\nSPLIT SUMMARY (processed):")
    for s in ("train","val","test"):
        c_hr = count(os.path.join(PROC_FOLDER, s, "HR"))
        c_lr = count(os.path.join(PROC_FOLDER, s, "LR"))
        print(f"  {s}: HR={c_hr}  LR={c_lr}")

def summarize_splits(proc_folder):
    """
    Count images in processed splits and create a small summary plot + CSV.
    Saves:
      - data/processed/split_summary.png
      - data/processed/split_counts.csv
    """
    splits = ("train", "val", "test")
    categories = ["HR", "LR"]
    counts = {s: {c: 0 for c in categories} for s in splits}

    for s in splits:
        for c in categories:
            d = os.path.join(proc_folder, s, c)
            if os.path.isdir(d):
                counts[s][c] = len([f for f in os.listdir(d) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff"))])
            else:
                counts[s][c] = 0

    # Print the counts
    print("\n=== FINAL SPLIT COUNTS ===")
    total_images = 0
    for s in splits:
        hrc = counts[s]["HR"]
        lrc = counts[s]["LR"]
        total_images += hrc  # HR count is the authoritative number
        print(f"  {s}: HR={hrc}  LR={lrc}")

    # Avoid division by zero
    total = max(1, total_images)

    # Prepare values for plotting: use HR counts as representative
    values = [counts[s]["HR"] for s in splits]
    labels = [f"{s} ({v} imgs)" for s, v in zip(splits, values)]
    percents = [100.0 * v / total for v in values]

    # Save CSV summary
    csv_path = os.path.join(proc_folder, "split_counts.csv")
    try:
        with open(csv_path, "w") as f:
            f.write("split,HR_count,LR_count,percent_of_total\n")
            for s, v, p in zip(splits, values, percents):
                f.write(f"{s},{counts[s]['HR']},{counts[s]['LR']},{p:.2f}\n")
        print(f"Saved split counts CSV -> {csv_path}")
    except Exception as e:
        print("Warning: could not save CSV:", e)

    # Create plot if matplotlib is available
    out_png = os.path.join(proc_folder, "split_summary.png")
    if plt is None:
        print("matplotlib not available — skipping plot generation.")
        return

    try:
        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(splits, values, color=["#4C72B0","#55A868","#C44E52"])
        ax.set_title("Dataset split (train / val / test)")
        ax.set_ylabel("Number of HR images")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2.0, v + max(1, total*0.01), str(v), ha='center', va='bottom')

        # add a small pie chart inset showing percentages
        from matplotlib.transforms import Bbox
        left, bottom, width, height = 0.65, 0.55, 0.3, 0.35
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.pie(values, labels=[f"{p:.1f}%" for p in percents], colors=["#4C72B0","#55A868","#C44E52"], autopct=None, startangle=90)
        ax2.set_title("Percent")

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved split summary plot -> {out_png}")
    except Exception as e:
        print("Warning: failed to create plot:", e)

def main():
    make_dirs()

    hr_files = list_images(HR_FOLDER)
    if not hr_files:
        raise SystemExit(f"No HR images found in {HR_FOLDER} — please add them and retry.")

    # create LR images for each HR file
    create_blurry_lr_files(hr_files)

    # copy into fixed splits using HR list order
    fixed_split_copy(hr_files)

    # --- NEW: summarize + plot the resulting splits ---
    summarize_splits(PROC_FOLDER)

    print("\nDone.")
    print("Check these folders:")
    print(" - intermediate LR:", LR_FOLDER)
    print(" - processed/train/HR and processed/train/LR")
    print(" - processed/val/HR and processed/val/LR")
    print(" - processed/test/HR and processed/test/LR")
    print(" - split summary (png/csv):", os.path.join(PROC_FOLDER, "split_summary.png"), os.path.join(PROC_FOLDER, "split_counts.csv"))

if __name__ == "__main__":
    main()