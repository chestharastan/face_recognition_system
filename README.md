Your README is already good 👍 — it just needs **grammar fixes, clearer explanations, and a cleaner structure** so it looks professional on GitHub.

Below is a **cleaned and improved version** while keeping your idea.

---

# Face Recognition System

This project captures a user's face from a camera and identifies the person using two different approaches:

* **Machine Learning** using Local Binary Pattern Histogram (LBPH)
* **Deep Learning** using a pre-trained **InceptionResNetV2** model

The **LBPH model** is faster and suitable for low-end systems but has limited accuracy.
The **deep learning model** provides higher accuracy and better robustness for real-world scenarios.

---

# How to Run This Project

## 1. Clone the Repository

```bash
git clone https://github.com/salarymakage/face_recognition_system.git
cd face_recognition_system
```

---

## 2. Set Up the Environment

### Windows

Create a virtual environment:

```bash
python -m venv env
```

Activate it:

**Command Prompt**

```bash
env\Scripts\activate
```

**PowerShell**

```bash
.\env\Scripts\Activate.ps1
```

---

### Mac or Linux

```bash
python3 -m venv env
source env/bin/activate
```

---

# Project Structure

```
Face_pp/
│
├── ml-lbph/                     # Machine Learning (LBPH) implementation
│   ├── train.py                 # Train LBPH model
│   ├── val.py                   # Validate model
│   └── webcam_test.py           # Real-time recognition with webcam
│
├── dl-resnetv2/                 # Deep Learning implementation
│   ├── train.py                 # Train deep learning model
│   ├── utils.py                 # Utility functions
│   └── webcam_test.py           # Real-time recognition with webcam
│
├── helpers/                     # Face detection modules
│   ├── dnn_detector.py          # DNN-based face detector
│   └── haar_detector.py         # Haar Cascade face detector
│
├── source/                      # Pre-trained detection models
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── haarcascade_frontalface_default.xml
│
├── capture.py                   # Capture images for dataset
├── split_dataset/               # Dataset for training and validation
└── README.md
```

---

# Machine Learning Approach (LBPH)

## Step 1: Capture Face Images

Before training the model, capture face images to build the dataset:

```bash
python capture.py
```

---

## Step 2: Prepare the Dataset

Organize the dataset as follows:

```
split_dataset/
│
├── train/
│   ├── Borom/
│   ├── Ketya/
│   ├── Norak/
│   ├── Thareah/
│   └── Thona/
│
└── validation/
    ├── Borom/
    ├── Ketya/
    ├── Norak/
    ├── Thareah/
    └── Thona/
```

Each folder should contain images of the corresponding person.

---

## Step 3: Train the Model

```bash
python ml-lbph/train.py
```

---

## Step 4: Test the Model

Run real-time recognition using the webcam:

```bash
python ml-lbph/webcam_test.py
```

---

# Deep Learning Approach (InceptionResNetV2)

The deep learning model uses the same dataset.

## Train the Model

```bash
python dl-resnetv2/train.py
```

---

## Test with Webcam

```bash
python dl-resnetv2/webcam_test.py
```

---

# Face Detection Methods

Two face detection methods are included:

* **Haar Cascade** (fast but less accurate)
* **DNN SSD Detector** (more accurate but slightly slower)

You can switch between detectors in the helper modules.

---

# Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Deep Learning (InceptionResNetV2)
* Machine Learning (LBPH)

---
