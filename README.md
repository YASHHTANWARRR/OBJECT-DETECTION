# 🧠 Object Detection System (YOLOv8 + OpenCV)

This project implements a real-time **object detection system** using **YOLOv8** and **OpenCV** to detect and classify multiple objects from images, videos, or webcam streams.

---

## 🚀 Features

* 🎯 Real-time object detection
* 🎥 Supports webcam, images, and video input
* 🧠 Uses **YOLOv8 (pretrained model)**
* ⚡ Fast inference with GPU support (CUDA)
* 🖼️ Bounding boxes with labels & confidence scores
* 📊 Detects multiple objects in a single frame

---

## 📂 Project Structure

```
OBJECT-DETECTION/
│── image_detect/
│── yolo_service.py
│── yolov8n.pt
│── README.md
```

---

## 🏷️ Classes

The YOLOv8 model is trained on the **COCO dataset**, which supports detection of 80+ object categories such as:

| Category | Examples              |
| -------- | --------------------- |
| People   | Person                |
| Vehicles | Car, Bus, Bike        |
| Animals  | Dog, Cat              |
| Objects  | Bottle, Chair, Laptop |

---

## ⚙️ Installation

### 1. Create Environment

```bash
conda create -n object_detect python=3.10 -y
conda activate object_detect
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

---

## 🧠 Model Architecture

* Model: **YOLOv8 Nano (yolov8n)**
* Pretrained on: **COCO dataset**
* Single-stage detector for fast real-time performance

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

---

## 🏃 Running the Project

```bash
python yolo_service.py
```

---

## 📊 Output

* 📦 Bounding boxes around detected objects
* 🏷️ Object labels
* 📈 Confidence scores
* 🎥 Real-time detection window

---

## 🧠 Key Learnings

* YOLO enables **real-time detection in a single forward pass**
* Pretrained models remove need for large datasets
* GPU acceleration significantly improves performance
* Computer vision pipelines require efficient frame processing

---

## ⚠️ Hardware Requirements

* CPU supported (slower performance)
* GPU recommended (CUDA enabled)
* Minimum 4GB RAM

---

## 📈 Future Improvements

* 🔥 Custom dataset training
* 🔥 Object tracking (DeepSORT)
* 🔥 Web app deployment (Flask / Streamlit)
* 🔥 Model optimization (TensorRT / ONNX)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

This project is for educational and research purposes.
