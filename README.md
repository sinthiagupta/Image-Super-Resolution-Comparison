📌 Project Title

Image Super-Resolution using SRCNN and Classical Upsampling Methods

📖 Overview

This project implements Single Image Super-Resolution (SISR) using:

🔹 1. Three Classical Interpolation Methods

Nearest Neighbor

Bilinear

Bicubic

🔹 2. Deep Learning Model (Improved SRCNN)

A residual SRCNN model trained using patch-based learning with tile-based inference.

🔹 3. Comparison with Advanced Pretrained Models

FSRCNN (Fast Super-Resolution CNN)

EDSR (Enhanced Deep Super-Resolution)

🎯 Project Motivation

Low-resolution images suffer from:

pixelation

blurred textures

loss of detail

Classical interpolation enlarges an image but cannot restore lost information.

Deep learning models like SRCNN/FSRCNN/EDSR learn how LR and HR relate and can reconstruct realistic textures.

This project builds a complete pipeline:

Preprocessing & dataset creation

Classical algorithm comparison

SRCNN training

Testing & evaluation

Comparison with state-of-the-art models

📂 Project Folder Structure
project/
│
├── data/
│   ├── HR/                     # Original high-resolution images
│   ├── LR/                     # Automatically generated LR images
│   └── processed/
│       ├── train/HR, LR        # 80 images
│       ├── val/HR, LR          # 10 images
│       └── test/HR, LR         # 10 images
│       ├── split_summary.png   # Train/Val/Test visualization
│       └── split_counts.csv    # Split counts
│
├── models/
│   └── srcnn_res_trained.h5    # Trained SRCNN model
│
├── outputs/
│   ├── images/                 # SRCNN LR/Pred/HR outputs
│   └── sample/
│       ├── nearest/            # Classical NN outputs
│       ├── bilinear/           # Classical Bilinear outputs
│       └── bicubic/            # Classical Bicubic outputs
│
├── scripts/
│   ├── preprocess_fixed_first80_81_90.py
│   ├── convert_make_lr_and_upscale_bicubic.py
│   ├── _train_srcnn.py
│   └── evaluate_metrics.py     # (optional evaluation script)
│
└── README.md

🗂 Dataset Preparation

Run:

python scripts/preprocess_fixed_first80_81_90.py


This script:

Reads HR images from data/HR/

Creates LR versions using bicubic downsampling

Splits into:

80 train

10 validation

10 test

Saves the split summary visualization (split_summary.png)

📈 Classical Upsampling Methods

Script:

python scripts/convert_make_lr_and_upscale_bicubic.py


This generates LR → HR outputs using:

🔹 Nearest Neighbor

Fastest, blocky output.

🔹 Bilinear

Smooth but blurry.

🔹 Bicubic

Best of the classical methods.

Outputs saved in:

outputs/sample/nearest/
outputs/sample/bilinear/
outputs/sample/bicubic/

🤖 Deep Learning Model — Residual SRCNN

We implement an improved SRCNN with:

64×64 patch-based training

Residual learning (predicts HR-LR difference)

Tiled inference to merge overlapping predictions smoothly

Adam optimizer, LR = 1e-4

Loss = MSE

🔧 Model Architecture
Input (64×64×3)
│
├── Conv2D (64 filters, 9×9, ReLU)
├── Conv2D (32 filters, 1×1, ReLU)
├── Conv2D (3 filters, 5×5)
│
└── Residual Add (input + conv3 output)

🏋️ Training Command
python scripts/_train_srcnn.py

🔍 Output

Predicted HR images saved in:

outputs/images/

📊 Evaluation Metrics

We evaluate 5 test images using:

✔ MSE — Mean Squared Error

Lower = better

✔ PSNR — Peak Signal-to-Noise Ratio

Higher (in dB) = sharper image

✔ SSIM — Structural Similarity Index

Higher (0–1) = more similar structure

Evaluation formulas follow standard methods and compute metrics on Y-channel (brightness) for accuracy.

🧪 Results (Your Actual Project Results)
🔹 Classical vs SRCNN (Average of 5 Test Images)
Method	Avg MSE	Avg PSNR (dB)	Avg SSIM
SRCNN (Your Model)	171.35	26.322	0.7786
Bicubic	175.13	26.230	0.7722
Bilinear	200.63	25.631	0.7456
Nearest	228.90	25.067	0.7381
🔹 Interpretation

SRCNN outperforms all classical methods

Bicubic is best classical

Nearest is worst due to block artifacts

🥇 Comparison with Pretrained Models (Benchmark Values)
Model	PSNR	SSIM	Dataset
SRCNN (Ours)	26.32	0.7786	Our test set
FSRCNN	~27.2	~0.81	Set5
EDSR	~32.4	~0.90	DIV2K
Why include EDSR & FSRCNN?

To show:

where modern SR networks stand

how deep learning evolved from SRCNN

how performance improves with deeper architectures

🚀 How to Run the Full Pipeline
1. Preprocess Data
python scripts/preprocess_fixed_first80_81_90.py

2. Generate Classical Upsampled Outputs
python scripts/convert_make_lr_and_upscale_bicubic.py

3. Train SRCNN
python scripts/_train_srcnn.py

4. Evaluate Metrics (optional)
python scripts/evaluate_metrics.py

🔮 Future Improvements

Integrate FSRCNN training

Add EDSR evaluation using pretrained weights

Use perceptual losses (VGG/LPIPS)

Add real-world degradation models (noise, JPEG artifacts)

Switch to GAN models (SRGAN, ESRGAN)

🏁 Conclusion

This project demonstrates:

How classical interpolation behaves

How SRCNN improves image quality

How modern deep SR networks outperform classical methods

It provides a complete and clear pipeline for learning, training, comparing, and evaluating image super-resolution methods.
