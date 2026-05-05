# Brain-Tumor-Detection-Using-Deep-Feature-And-Machine-Learning

A machine learning and deep learning project to detect and classify brain tumors from MRI scan images using multiple classification algorithms.

---

## Project Overview

Brain tumor detection is a critical medical imaging task. This project automates the classification of brain MRI scans into four categories using both traditional Machine Learning models and a Convolutional Neural Network (CNN).

**Classes:**
- `no_tumor` - No tumor detected
- `glioma_tumor` - Glioma tumor
- `meningioma_tumor` - Meningioma tumor
- `pituitary_tumor` - Pituitary tumor

---

## Dataset

- **Source:** Brain Tumor MRI Dataset (stored on Google Drive)
- **Structure:**
```
Brain_Tumor_Dataset/
├── Training/
│   ├── no_tumor/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── no_tumor/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    └── pituitary_tumor/
```
- **Image Size:** Resized to 224×224 pixels (grayscale)
- **Class Balancing:** Applied `RandomOverSampler` to handle class imbalance

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Machine Learning | Scikit-learn, XGBoost |
| Image Processing | OpenCV (cv2), PIL |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab |

---

## Project Workflow

```
MRI Images → Preprocessing → Feature Extraction → Model Training → Evaluation → Prediction
```

1. **Data Loading** — MRI images loaded from Google Drive in grayscale
2. **Preprocessing** — Resize to 224×224, flatten, normalize with StandardScaler
3. **Class Balancing** — RandomOverSampler applied to balance dataset
4. **Train-Test Split** — 80% training, 20% testing
5. **Model Training** — 8 models trained and evaluated
6. **Evaluation** — Accuracy score, classification report, confusion matrix
7. **Prediction** — Visual prediction output on test images

---

## Models Implemented

### Machine Learning Models

| # | Model | Description |
|---|---|---|
| 1 | Support Vector Machine (SVM) | Kernel-based classifier |
| 2 | Gaussian Naive Bayes | Probabilistic classifier |
| 3 | Decision Tree | Rule-based tree classifier |
| 4 | Random Forest | Ensemble of 100 decision trees |
| 5 | K-Nearest Neighbors (KNN) | Distance-based classifier (k=3) |
| 6 | Logistic Regression | Linear probabilistic model |
| 7 | XGBoost | Gradient boosting classifier |
| 8 | MLP Classifier | Neural network (100 hidden units, 100 epochs) |

### Deep Learning Model

| # | Model | Architecture |
|---|---|---|
| 9 | CNN | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(4, softmax) |

---

## CNN Architecture

```
Input (224×224×1)
   ↓
Conv2D (32 filters, 3×3, ReLU)
   ↓
MaxPooling2D (2×2)
   ↓
Conv2D (64 filters, 3×3, ReLU)
   ↓
MaxPooling2D (2×2)
   ↓
Flatten
   ↓
Dense (128, ReLU)
   ↓
Dense (4, Softmax)
```

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 100
- **Data Augmentation:** Shear, Zoom, Horizontal Flip

---

## Evaluation Metrics

Each model is evaluated using:
- Training & Testing Accuracy
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix (visualized with Seaborn heatmap)
- Training vs Validation Accuracy/Loss curves (CNN & MLP)

---

## How to Run

1. **Open in Google Colab**
2. **Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. **Upload your dataset** to Google Drive under:
```
My Drive/Brain_Tumor_Dataset/
```
4. **Run all cells** in order from top to bottom.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
opencv-python
tensorflow
keras
Pillow
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn opencv-python tensorflow Pillow
```

---

## Author

**Shavala Sundar Raju**  
B.Tech in Computer Science and Engineering  
Madanapalle Institute of Technology and Science (2021–2025)

---

## License

This project is for academic and educational purposes only.
