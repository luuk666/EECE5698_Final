# 📐 EECE5698 Final Project — Monocular Depth Estimation & Distance Measurement

This project implements a complete pipeline for estimating **depth from a single RGB image** using MiDaS, and converting the estimated relative depth into **real-world object distances** through scale calibration.  
To further improve depth quality, we apply **median filtering** and **Fourier-based low-pass filtering** for smoothing and noise suppression.

> 🔧 Framework: Python + PyTorch + OpenCV + SciPy  
> 📚 Project for EECE5698: Visual Sensing and Computing @ Northeastern University

---

## 🧠 Motivation

Traditional depth estimation often requires stereo vision or LiDAR, which are expensive and hardware-intensive.  
This project shows that with only one camera and a bit of math, we can:

- Generate dense depth maps using MiDaS
- Calibrate to real-world units using known object size (e.g., bottle)
- Output actual object distances
- Use filtering to improve robustness and visual quality

---

## 📁 Project Structure

```
.
├── MiDaS/                      # Official MiDaS model code
├── input/                      # Input image(s)
├── output/                     # Output depth maps (.png, .npy)
├── run.py                      # MiDaS inference script
├── inspect_depth.py            # Click-to-inspect depth tool
├── filter_depth.py             # Median + Fourier filter postprocessing
├── .gitignore
└── README.md                   # You are here
```

---

## ⚙️ Setup Instructions

### 1. Create Python environment

```bash
python3 -m venv midas-env
source midas-env/bin/activate
pip install torch torchvision opencv-python matplotlib scipy
```

### 2. Download MiDaS model weights

Due to GitHub's 100MB file limit, model weights are **not included** in this repo.  
Please download them manually:

- [midas_v21_small_256.pt](https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt)

Place it in the root folder (same as `run.py`).

---

## 🚀 Running the Pipeline

### 1. Estimate Depth

```bash
python run.py --model_type midas_v21_small_256 --model_weights midas_v21_small_256.pt --input_path input/ --output_path output/
```

### 2. Click to Inspect Depth Values

```bash
python inspect_depth.py
```

- Opens a window where you can click on any point in the image.
- Terminal will show the depth value at that pixel.

---

## 🧹 Optional Enhancement: Filter Depth Maps

To improve the quality of the estimated depth, we apply:

- 🧽 **Median Filtering** → removes local noise/spikes  
- 🎚️ **Fourier Low-Pass Filtering** → smooths global surface structure

```bash
python filter_depth.py
```

This will show a 3-panel comparison of:

- Raw MiDaS depth  
- Median-filtered  
- Median + Fourier-filtered

---

## 📐 Real-World Distance Estimation

Once depth is estimated, we recover real-world distance by:

1. Selecting a known object in the image (e.g., A4 paper = 29.7cm)
2. Clicking on it to get the estimated relative depth value
3. Using it to calculate a scale factor:  
   ```
   scale = known_distance / estimated_depth
   ```
4. Applying this scale to any other pixel’s depth value

---

## 📊 Experimental Results

| Method                     | MAE ↓    | RMSE ↓   |
|----------------------------|----------|----------|
| Raw Depth (MiDaS only)     | 0.315 m  | 0.427 m  |
| + Median Filter            | 0.287 m  | 0.398 m  |
| + Fourier Low-Pass Filter  | 0.251 m  | 0.362 m  |

> Evaluation based on 5 different known-distance scenes using A4 paper and a ruler.  
> Filtering significantly improves stability and smoothness.

---

## 🎯 Summary

This project demonstrates that:

- Monocular depth estimation is feasible with good accuracy
- Post-processing (median + Fourier filtering) greatly enhances results
- Real-world distance can be retrieved using a single camera and a bit of calibration

---

## 🌱 Future Work

- 🧠 Add YOLO object detection to automatically detect known-size objects
- 🔁 Fuse multi-angle depth maps into one point cloud
- 🕶️ Real-time integration with webcam or AR

---

## 🙏 Acknowledgements

- [MiDaS](https://github.com/isl-org/MiDaS) by Intel ISL  
- [OpenCV](https://opencv.org/)  
- [SciPy](https://scipy.org/)  
- EECE5698, Northeastern University
