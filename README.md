# Face Recognition — JKT48 Member Identification (CNN + PCA + SVM)

A multi-class face identification system for recognizing 40 JKT48 members under varying pose and lighting conditions, using a combination of **EfficientNetV2-S** (feature extractor), **PCA** (dimensionality reduction), and **SVM** (classification). This project is an implementation of the undergraduate thesis *"Face Image Detection as an Identifier Using Convolutional Neural Network and Support Vector Machine Methods."*

## 🎯 Background

**JKT48** is a Jakarta-based idol group and the official sister group of Japan's AKB48, consisting of dozens of female members who perform in regular theater shows, concerts, and various public media. With a large and frequently changing lineup, plus high visual similarity between members (similar face shapes, hairstyles, and uniform costumes), automatically identifying individual members from images is a notable challenge — especially in the music entertainment industry during live performances, where variations in stage lighting, pose, and expression often cause conventional identification systems to fail at recognizing individuals accurately.

Identifying individuals within a large group with high visual similarity (face shape, hairstyle, expression) is a classic computer vision challenge, made harder by pose and lighting variation in stage/public photos. This project tests whether combining a modern CNN with classical dimensionality reduction (PCA) and a classical classifier (SVM) can achieve high accuracy **while remaining computationally efficient** — compared against classical approaches (Eigenfaces, Fisherfaces) and pure deep learning models (AlexNet, ResNet18, VGGNet16).

## 📊 Dataset

- **8,000 face images** of **40 JKT48 members**, collected from public sources and manually annotated.
- Each class includes variation across conditions: *frontal-bright*, *frontal-dark*, *non-frontal-bright*, *non-frontal-dark*.
- The dataset went through **stratified splitting** (train/val), **face alignment** (MTCNN), and **augmentation** (horizontal flip, ±15° rotation).

## 🔧 Pipeline

1. **EDA** — exploring class distribution, pose & lighting variation, image resolution.
2. **Stratified Split** — splitting the dataset per class into train/validation sets (25% validation ratio).
3. **Face Alignment & Augmentation** — face detection and alignment with MTCNN, augmentation with Albumentations.
4. **Feature Extraction** — EfficientNetV2-S (without pretrained weights) producing 1,280-dimensional feature vectors.
5. **Dimensionality Reduction** — PCA with cumulative variance thresholds of 90% / 95% / 99%.
6. **Classification** — SVM with RBF kernel.
7. **Hyperparameter Tuning** — grid search across 45 combinations (PCA components × C × gamma).
8. **Benchmarking** — compared against Eigenfaces, Fisherfaces, AlexNet, ResNet18, and VGGNet16 (each with & without PCA+SVM).
9. **Inference Time & Model Size Analysis** — evaluating deployment feasibility.

## 🏆 Main Model Results

**EfficientNetV2-S (no pretrained weights) + PCA (99%) + SVM (RBF)**

| Metric | Value |
|---|---|
| Accuracy | **80.18%** |
| Precision | 80.43% |
| Recall | 80.20% |
| F1-score | 80.20% |
| Inference time | 0.0303 s/image (~33 images/second) |
| Model size | 86.13 MB |

Applying PCA cut inference time from **0.2306 seconds** (without PCA) down to **0.0303 seconds**, with no meaningful drop in accuracy (80.22% → 80.18%).

## 📈 Model Comparison

| Model | Accuracy | Inference Time (s) | Model Size (MB) |
|---|---|---|---|
| **EfficientNetV2-S (no pretrained) + PCA + SVM** ⭐ | **80.18%** | 0.0303 | 86.13 |
| EfficientNetV2-S (no pretrained) + SVM (no PCA) | 80.22% | 0.2307 | 136.84 |
| EfficientNetV2-S (pretrained) + PCA + SVM | 56.27% | 0.0331 | 133.21 |
| ResNet18 (pretrained) | 92.57% | 0.0036 | 42.79 |
| VGGNet16 (pretrained) | 87.17% | 0.0024 | 512.80 |
| AlexNet (pretrained) | 83.27% | 0.0007 | 218.08 |
| Eigenfaces (PCA + SVM) | 30.48% | 0.0191 | 126.23 |
| Fisherfaces (PCA + LDA + SVM) | 29.68% | 0.0034 | 70.70 |

> Pretrained deep learning models (ResNet18, VGGNet16, AlexNet) achieve higher raw accuracy, but come with much larger model sizes (>200 MB for AlexNet and VGGNet16), making them less ideal for resource-constrained devices. The main model in this project was chosen for offering the **best balance** between accuracy (~80%), inference speed, and model size (86 MB) — without relying on ImageNet transfer learning.

## 🔬 Hyperparameter Tuning

A grid search was run across 45 parameter combinations:

| Parameter | Values Tested |
|---|---|
| PCA components | 90% (17 components), 95% (30 components), 99% (93 components) |
| C (regularization) | 0.1 / 1 / 10 |
| Gamma (kernel coefficient) | 'scale' / 0.001 / 0.01 / 0.1 / 1 |

Best model: **PCA 99% + C=10 + gamma='scale'**, achieving 99.97% cross-validation accuracy and 80.18% test accuracy.

## ⚠️ Limitations

- The model still struggles to distinguish faces with high visual similarity (e.g., the *Ella* class only reached an F1-score of 56%, while the *Elin* class reached 94%).
- Performance drops under extreme lighting conditions or uneven representation of non-frontal poses in certain classes.
- Not yet fully real-time at large scale (60 FPS).

## 🛠️ Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` (PCA, SVM, LDA) · `OpenCV` · `MediaPipe` · `MTCNN` · `Albumentations` · `Pandas` · `Matplotlib` / `Seaborn`

## 📁 Notebook Structure

```
final_script.ipynb
├── 1. EDA
├── 2. Stratified Dataset Split
├── 3. Face Alignment & Augmentation
├── 4. EfficientNetV2-S (no pretrained)
├── 5. PCA + SVM
├── 6. Hyperparameter Tuning
├── 7. Best Model Evaluation
├── 8. EffNetV2-S without PCA (baseline)
├── 9. EfficientNetV2-S (pretrained)
├── 10. EfficientNetV2-S (pretrained) + PCA + SVM
├── 11. Eigenfaces
├── 12. Fisherfaces
├── 13. AlexNet (+ PCA + SVM)
├── 14. ResNet18 (+ PCA + SVM)
├── 15. VGGNet16 (+ PCA + SVM)
└── 16. Inference Time Benchmark
```

## 🚀 Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset**
   The face image dataset (8,000 images of 40 JKT48 members) is hosted on Google Drive since it's too large for this repo:
   👉 [Download dataset from Google Drive](<google-drive-dataset-link>)

   Extract it locally, e.g. into a `data/` folder in the project root.

3. **Update the file paths in the notebook**
   Open `final_script.ipynb` and update the dataset/output path variables to match your local machine (they currently point to the original author's local paths, e.g. `D:/Kuliah/Skripsi/...`). Look for variables such as:
   ```python
   root_folder = "D:/Kuliah/Skripsi/Data"                    # → change to your dataset path
   dataset_path = Path("D:/Kuliah/Skripsi/Data")
   output_dir = Path("D:/Kuliah/Skripsi/Data_StratifiedSplit")
   input_root = Path("D:/Kuliah/Skripsi/Data_StratifiedSplit")
   output_root = Path("D:/Kuliah/Skripsi/Data_Preprocessed")
   train_dir = "D:/Kuliah/Skripsi/Data_Preprocessed/train"
   val_dir   = "D:/Kuliah/Skripsi/Data_Preprocessed/val"
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook final_script.ipynb
   ```
   Run the cells in order, starting from EDA through model training/evaluation, based on which architecture you want to reproduce.

5. **(Optional) Skip training — use the pretrained model**
   Training all architectures from scratch (especially the deep learning models) can be heavy on time and compute. If you just want to try inference without retraining, a ready-to-use trained model (weights + PCA + SVM pipeline) is provided here:
   👉 [Download pretrained model](<google-drive-model-link>)

   Place the downloaded files in a `models/` folder, then run only the **Inference / Best Model Testing** section of the notebook (skip the training cells) to load the model directly and run predictions.

**Main dependencies:** `torch`, `torchvision`, `opencv-python`, `mediapipe`, `mtcnn`, `albumentations`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `joblib`

## 📄 Reference

Full thesis: *Face Image Detection as an Identifier Using Convolutional Neural Network and Support Vector Machine Methods* — Arya Pramudika, Mathematics Study Program.

## 👤 Author

**Arya Pramudika**
