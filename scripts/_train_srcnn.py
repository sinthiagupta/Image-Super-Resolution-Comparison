# scripts/train_srcnn_now.py
"""
Train residual SRCNN on existing processed dataset and save inference outputs for the VAL set.
Assumes:
  data/processed/train/LR, data/processed/train/HR
  data/processed/val/LR,   data/processed/val/HR
Outputs saved to outputs/images/val_XXX_pred.png (and lr/hr).
"""

import os, random
import numpy as np
from PIL import Image

# robust keras import (works with tensorflow.keras or standalone keras)
try:
    import keras
    from keras import layers, models, optimizers
except Exception:
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers

# ---------------- CONFIG ----------------
PATCH = 64              # patch size used to train (patches come from full-size images)
EPOCHS = 4
BATCH = 20
STEPS_PER_EPOCH = 120   # lower if your CPU is slow (e.g. 50)
VAL_STEPS = 40
LEARNING_RATE = 1e-4

TRAIN_LR_DIR = "data/processed/train/LR"
TRAIN_HR_DIR = "data/processed/train/HR"
VAL_LR_DIR   = "data/processed/val/LR"
VAL_HR_DIR   = "data/processed/val/HR"

MODEL_PATH = "models/srcnn_res_trained.h5"
OUT_DIR = "outputs/images"
os.makedirs("models", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------------------

# ---------- helpers ----------
def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])

def sample_patch_pair(lr_path, hr_path, patch_size):
    lr = Image.open(lr_path).convert("RGB")
    hr = Image.open(hr_path).convert("RGB")
    w, h = lr.size
    if w < patch_size or h < patch_size:
        # pad/resize minimally so patch is possible
        lr = lr.resize((max(w,patch_size), max(h,patch_size)), Image.BICUBIC)
        hr = hr.resize((max(w,patch_size), max(h,patch_size)), Image.BICUBIC)
        w, h = lr.size
    x = random.randint(0, w - patch_size)
    y = random.randint(0, h - patch_size)
    lr_patch = np.array(lr.crop((x,y,x+patch_size,y+patch_size)), dtype=np.float32) / 255.0
    hr_patch = np.array(hr.crop((x,y,x+patch_size,y+patch_size)), dtype=np.float32) / 255.0
    return lr_patch, hr_patch

def patch_generator(lr_dir, hr_dir, lr_files, hr_files, batch_size, patch_size):
    n = min(len(lr_files), len(hr_files))
    if n == 0:
        raise RuntimeError("No image pairs found in " + lr_dir)
    while True:
        Xb = []
        Yb = []
        for _ in range(batch_size):
            i = random.randrange(n)
            lr_path = os.path.join(lr_dir, lr_files[i])
            hr_path = os.path.join(hr_dir, hr_files[i])
            lp, hp = sample_patch_pair(lr_path, hr_path, patch_size)
            Xb.append(lp); Yb.append(hp)
        yield np.stack(Xb, axis=0), np.stack(Yb, axis=0)

# --------------- model ---------------
def build_residual_srcnn(patch):
    inp = layers.Input((patch, patch, 3))
    x = layers.Conv2D(64, (9,9), activation='relu', padding='same')(inp)
    x = layers.Conv2D(32, (1,1), activation='relu', padding='same')(x)
    res = layers.Conv2D(3, (5,5), activation='linear', padding='same')(x)
    out = layers.Add()([inp, res])   # residual connection
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='mse')
    return model

# ------------ tile inference -------------
def tile_predict_full(model, lr_img, patch, overlap):
    lr = np.array(lr_img.convert("RGB"), dtype=np.float32) / 255.0
    H, W, C = lr.shape
    out = np.zeros_like(lr)
    weight = np.zeros((H, W, 1), dtype=np.float32)
    step = patch - overlap
    if step <= 0:
        step = patch // 2
    for y in range(0, H, step):
        for x in range(0, W, step):
            x0 = x; y0 = y
            x1 = min(x0 + patch, W)
            y1 = min(y0 + patch, H)
            tile = lr[y0:y1, x0:x1, :]
            ph, pw = tile.shape[:2]
            if ph != patch or pw != patch:
                pad_h = patch - ph; pad_w = patch - pw
                tile = np.pad(tile, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')
            inp = np.expand_dims(tile, 0)
            pred = model.predict(inp, verbose=0)[0]
            pred = np.clip(pred, 0.0, 1.0)
            pred = pred[:ph, :pw, :]
            out[y0:y1, x0:x1, :] += pred
            weight[y0:y1, x0:x1, 0] += 1.0
    weight[weight==0] = 1.0
    out = out / weight
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

# --------------- main flow ----------------
def main():
    random.seed(1234)
    np.random.seed(1234)

    # check data exists
    train_lr_files = list_images(TRAIN_LR_DIR)
    train_hr_files = list_images(TRAIN_HR_DIR)
    val_lr_files = list_images(VAL_LR_DIR)
    val_hr_files = list_images(VAL_HR_DIR)

    if len(train_lr_files) == 0 or len(train_hr_files) == 0:
        raise SystemExit("Train LR/HR not found. Check data/processed/train/")
    if len(val_lr_files) == 0 or len(val_hr_files) == 0:
        raise SystemExit("Val LR/HR not found. Check data/processed/val/")

    print("Train pairs:", len(train_lr_files), " Val pairs:", len(val_lr_files))

    model = build_residual_srcnn(PATCH)
    model.summary()

    train_gen = patch_generator(TRAIN_LR_DIR, TRAIN_HR_DIR, train_lr_files, train_hr_files, BATCH, PATCH)
    val_gen = patch_generator(VAL_LR_DIR, VAL_HR_DIR, val_lr_files, val_hr_files, BATCH, PATCH)

    print("Training...")
    model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=VAL_STEPS
    )

    model.save(MODEL_PATH)
    print("Saved model to", MODEL_PATH)

    # Run inference on VAL set and save outputs
    print("Running inference on VAL set and saving outputs to", OUT_DIR)
    overlap = max(8, PATCH//4)
    n = min(len(val_lr_files), len(val_hr_files))
    for i in range(n):
        lr_path = os.path.join(VAL_LR_DIR, val_lr_files[i])
        hr_path = os.path.join(VAL_HR_DIR, val_hr_files[i])
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        pred_arr = tile_predict_full(model, lr_img, PATCH, overlap)
        prefix = f"val_{i+1:03d}"
        Image.fromarray(np.array(lr_img)).save(os.path.join(OUT_DIR, f"{prefix}_lr.png"))
        Image.fromarray(pred_arr).save(os.path.join(OUT_DIR, f"{prefix}_pred.png"))
        hr_img.save(os.path.join(OUT_DIR, f"{prefix}_hr.png"))
        print("Saved:", prefix)
    print("All done. Outputs in", OUT_DIR)

if __name__ == "__main__":
    main()
