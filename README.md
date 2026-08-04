# Face Mask Recognition using CNN and OpenCV

A real-time face mask detection system built using TensorFlow, Keras, and OpenCV.

## Features

- Real-time face detection using Haar Cascade
- CNN-based mask classification
- Detects:
  - Mask
  - No Mask
- Uses webcam/mobile camera input through Camo Studio
- Displays prediction confidence

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- CNN (Convolutional Neural Network)

## Project Structure
Face Mask Recognition/
│
├── datasets/
│ ├── train/
│ ├── valid/
│ └── test/
│
├── models/
│ └── mask_detector.keras
│
├── haarcascade_frontalface_default.xml
├── train.py
├── detect_mask.py
└── README.md

## Model Performance

- Test Accuracy: **92.19%**

## How to Run

### 1. Activate virtual environment

Windows:
venv\Scripts\activate

### 2. Install dependencies
pip install tensorflow opencv-python numpy

### 3. Train the model
python train.py

### 4. Run real-time detection
python detect_mask.py

## Demo

The system detects faces from the camera feed and displays:

- Mask ✅
- No Mask ❌

## Author

Vipin Yadav