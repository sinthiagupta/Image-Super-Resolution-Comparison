# scripts/1_prepare_simple.py
# Very simple: HR → LR + simple preprocessing + split

import os
from PIL import Image
import random
from shutil import copyfile

# FOLDERS
ROOT = os.path.dirname(os.path.dirname(__file__))    # repo folder
HR_FOLDER = os.path.join(ROOT, "data", "HR")
LR_FOLDER = os.path.join(ROOT, "data", "LR")         # temporary LR output
PROC_FOLDER = os.path.join(ROOT, "data", "processed")

# SETTINGS
SCALE = 4   # HR → LR scale
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def make_dirs():
    os.makedirs(LR_FOLDER, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(PROC_FOLDER, split, "LR"), exist_ok=True)

def convert_hr_to_lr():
    print("Converting HR → LR ...")
    files = sorted(os.listdir(HR_FOLDER))

    for f in files:
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        hr_path = os.path.join(HR_FOLDER, f)
        lr_path = os.path.join(LR_FOLDER, f.replace(".png", "_lr.png"))

        img = Image.open(hr_path).convert("RGB")     # simple preprocess
        w, h = img.size
        img_lr = img.resize((w//SCALE, h//SCALE), Image.BICUBIC)
        img_lr.save(lr_path)

    print("LR images saved in data/LR")

def split_data():
    print("Splitting dataset...")
    lr_files = sorted(os.listdir(LR_FOLDER))
    random.shuffle(lr_files)

    n = len(lr_files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {
        "train": lr_files[:n_train],
        "val": lr_files[n_train:n_train+n_val],
        "test": lr_files[n_train+n_val:],
    }

    for split, flist in splits.items():
        for f in flist:
            src = os.path.join(LR_FOLDER, f)
            dst = os.path.join(PROC_FOLDER, split, "LR", f)
            copyfile(src, dst)

    print("Done! LR dataset split into train/val/test.")

def main():
    make_dirs()
    convert_hr_to_lr()
    split_data()

if __name__ == "__main__":
    main()
