# 🌿 Plant Disease Classification — MobileNetV2

A deep learning model for classifying **38 plant disease conditions** across 14 crop types using **Transfer Learning with MobileNetV2**. The project covers the full ML pipeline: data preprocessing, augmentation, model training, evaluation, Grad-CAM explainability, and TFLite mobile deployment.

---

## ✨ Features

- 🧠 **Transfer Learning** — MobileNetV2 pretrained on ImageNet with frozen base layers
- 📊 **38 Disease Classes** across 14 crops (Apple, Tomato, Corn, Grape, Potato, and more)
- 🔍 **Grad-CAM Visualization** — Heatmaps showing which regions the model focuses on
- 📱 **TFLite Export** — Converted and tested for mobile/edge deployment
- 📈 **Full Training Curves** — Accuracy and loss plots across all 25 epochs
- 🗂️ **Confusion Matrix** + Classification Report for full evaluation
- 💾 **Best Model Checkpoint** — Automatically saves best `val_accuracy` epoch

---

## 🏗️ Model Architecture

```
Input (224 × 224 × 3)
        ↓
MobileNetV2 Base — frozen (ImageNet weights)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (38, softmax)
        ↓
Output — 38 class probabilities
```

**Why GlobalAveragePooling2D over Flatten?**
- Fewer parameters
- Less overfitting
- Spatially more robust

---

## 📊 Dataset

| Split | Images |
|-------|--------|
| Train | ~34,741 (64%) |
| Validation | ~8,686 (16%) |
| Test | ~10,857 (20%) |
| **Total** | **~54,284** |

**Data Augmentation applied on training set:**

| Technique | Value |
|-----------|-------|
| Rotation | ±25° |
| Width / Height Shift | 25% |
| Shear | 0.2 |
| Zoom | 0.2 |
| Horizontal Flip | ✅ |
| Fill Mode | nearest |

---

## 📈 Training Results (25 Epochs)

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 79.45% | 89.76% | 0.7171 | 0.3271 |
| 5 | 90.15% | 93.85% | 0.3035 | 0.1927 |
| 10 | 90.68% | 94.14% | 0.2960 | 0.1893 |
| 21 | 90.87% | **94.62%** | 0.2904 | **0.1718** |
| 25 | 90.99% | 94.27% | 0.2879 | 0.1860 |

> ✅ Best model saved at **Epoch 21** — Val Accuracy: **94.62%**

---

## 🌿 Supported Classes (38 Total)

| Crop | Conditions |
|------|-----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Haunglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps that highlight the **exact regions** the model uses to make its prediction.

```
Original Image → Grad-CAM Heatmap → Overlay
```

- Uses the last conv layer `out_relu` from MobileNetV2
- Displays original image alongside heatmap overlay
- Shows true label vs predicted label with confidence %

---

## 📱 TFLite Deployment

The model is exported to TensorFlow Lite for mobile and edge devices.

**Model specs:**

| Property | Value |
|----------|-------|
| Input Shape | (1, 224, 224, 3) |
| Input Type | float32 |
| Input Range | [0, 1] |
| Output Shape | (1, 38) |
| Output Type | float32 (softmax probabilities) |

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn Pillow scikit-image
```

### Predict with Keras Model

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model('best_plant_model.h5')

img = load_img('your_image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

### Predict with TFLite Model

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = load_img('your_image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
confidence = np.max(output_data)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Framework | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet) |
| Language | Python 3.7+ |
| Data | Pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Image Processing | Pillow, scikit-image |
| Explainability | Grad-CAM (custom implementation) |
| Deployment | TensorFlow Lite |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
plant-disease-classification/
├── mobv2.ipynb               ← Full training & evaluation notebook
├── best_plant_model.h5       ← Saved Keras model (best val_accuracy)
├── model.tflite              ← TFLite converted model for mobile
└── README.md
```

---

## 📦 Code Files

<!-- Add your project files here (mobv2.ipynb, best_plant_model.h5, model.tflite) -->
