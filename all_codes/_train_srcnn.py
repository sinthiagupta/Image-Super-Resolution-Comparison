# scripts/_train_srcnn.py
"""
Train residual SRCNN on existing processed dataset and save inference outputs for the TEST set.
This version fixes LR->HR filename mismatches (e.g. LR names like "0091_lr.png" vs HR "0091.png")
"""

import os, random, sys
import numpy as np
from PIL import Image

# robust keras import
try:
    import keras
    from keras import layers, models, optimizers
except Exception:
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers

# ---------------- CONFIG ----------------
PATCH = 64
EPOCHS = 4
BATCH = 20
STEPS_PER_EPOCH = 120
VAL_STEPS = 40
LEARNING_RATE = 1e-4

TRAIN_LR_DIR = "data/processed/train/LR"
TRAIN_HR_DIR = "data/processed/train/HR"
VAL_LR_DIR   = "data/processed/val/LR"
VAL_HR_DIR   = "data/processed/val/HR"

TEST_LR_DIR  = "data/processed/test/LR"
TEST_HR_DIR  = "data/processed/test/HR"

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

# ---------- robust LR->HR matching ----------
def normalize_name(name):
    """
    Normalize a filename by removing common LR suffixes and extension,
    and also removing leading zeros for loose matching.
    """
    base = os.path.splitext(name)[0]
    # drop common suffixes
    for suf in ("_lr","-lr","_LR","-LR","_small","-small","_bicubic","_nearest","_pred","_hr"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    # strip leading zeros to be forgiving
    stripped = base.lstrip("0")
    if stripped == "":
        stripped = base  # if name was "000" keep original
    return base, stripped

def find_matching_hr(lr_name, hr_list):
    """
    Try multiple matching strategies to find hr filename for a given lr_name:
      1) exact same filename in hr_list
      2) remove common suffixes (0091_lr -> 0091) and try with common extensions
      3) match by base (strip prefixes/leading zeros)
      4) startswith match (relaxed)
    Returns HR filename (from hr_list) or None.
    """
    # 1) exact
    if lr_name in hr_list:
        return lr_name

    base, stripped = normalize_name(lr_name)

    # 2) try base with extensions
    for ext in (".png", ".jpg", ".jpeg"):
        cand = base + ext
        if cand in hr_list:
            return cand
        cand2 = stripped + ext
        if cand2 in hr_list:
            return cand2

    # 3) try matching by comparing normalized stripped base of hr files
    for h in hr_list:
        hb = os.path.splitext(h)[0]
        hb_norm = hb
        # drop common suffixes from hr too
        for suf in ("_hr","-hr","_HR","-HR"):
            if hb_norm.endswith(suf):
                hb_norm = hb_norm[:-len(suf)]
                break
        if hb_norm == base or hb_norm == stripped:
            return h
        if hb_norm.lstrip("0") == stripped:
            return h

    # 4) fallback: startswith (very relaxed)
    for h in hr_list:
        hb = os.path.splitext(h)[0]
        if hb.startswith(base) or base.startswith(hb) or hb.lstrip("0").startswith(stripped):
            return h

    return None

# --------------- main flow ----------------
def main():
    random.seed(1234)
    np.random.seed(1234)

    # check data exists
    train_lr_files = list_images(TRAIN_LR_DIR)
    train_hr_files = list_images(TRAIN_HR_DIR)
    val_lr_files = list_images(VAL_LR_DIR)
    val_hr_files = list_images(VAL_HR_DIR)

    test_lr_files = list_images(TEST_LR_DIR)
    test_hr_files = list_images(TEST_HR_DIR)

    if len(train_lr_files) == 0 or len(train_hr_files) == 0:
        raise SystemExit("Train LR/HR not found. Check data/processed/train/")
    if len(val_lr_files) == 0 or len(val_hr_files) == 0:
        raise SystemExit("Val LR/HR not found. Check data/processed/val/")

    print("Train pairs:", len(train_lr_files), " Val:", len(val_lr_files), " Test:", len(test_lr_files))

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

    # save model (HDF5 legacy warning is ok)
    try:
        model.save(MODEL_PATH)
        print("Model saved →", MODEL_PATH)
    except Exception as e:
        print("Warning: failed to save as HDF5:", e)
        alt = MODEL_PATH + ".keras"
        model.save(alt)
        print("Saved instead to", alt)

    # Run inference on TEST set and save outputs
    print("Running inference on TEST set and saving outputs...")
    overlap = max(8, PATCH//4)

    if len(test_lr_files) == 0:
        print("No test LR files found; nothing to do.")
        return

    # create a quick set of HR filenames for matching
    hr_set = set(test_hr_files)

    unmatched = []
    saved_count = 0

    for i, lr_fname in enumerate(test_lr_files):
        lr_path = os.path.join(TEST_LR_DIR, lr_fname)

        matched_hr = find_matching_hr(lr_fname, test_hr_files)
        if matched_hr is None:
            print("NO HR FOUND for LR:", lr_fname)
            unmatched.append(lr_fname)
            continue

        hr_path = os.path.join(TEST_HR_DIR, matched_hr)

        # Load images
        try:
            lr_img = Image.open(lr_path).convert("RGB")
        except Exception as e:
            print("ERROR opening LR:", lr_path, e)
            unmatched.append(lr_fname)
            continue

        try:
            hr_img = Image.open(hr_path).convert("RGB")
        except Exception as e:
            print("ERROR opening HR:", hr_path, e)
            unmatched.append(lr_fname)
            continue

        pred_arr = tile_predict_full(model, lr_img, PATCH, overlap)

        # Use matched_hr base for output prefix if possible, else fallback to test_i
        base_name = os.path.splitext(matched_hr)[0]
        safe_base = base_name
        prefix = f"{safe_base}_pred"

        out_lr_name = f"{safe_base}_lr.png"
        out_pred_name = f"{safe_base}_pred.png"
        out_hr_name = f"{safe_base}_hr.png"

        Image.fromarray(np.array(lr_img)).save(os.path.join(OUT_DIR, out_lr_name))
        Image.fromarray(pred_arr).save(os.path.join(OUT_DIR, out_pred_name))
        hr_img.save(os.path.join(OUT_DIR, out_hr_name))

        saved_count += 1
        print(f"Saved: {out_pred_name}  (matched HR: {matched_hr})")

    print(f"Done. Saved {saved_count} predicted images to {OUT_DIR}")
    if unmatched:
        print("Unmatched LR files (no HR found):", unmatched)
        print("Please check your data/processed/test/HR filenames and ensure matching names.")
    else:
        print("All test LR files had matching HRs.")

if __name__ == "__main__":
    main()