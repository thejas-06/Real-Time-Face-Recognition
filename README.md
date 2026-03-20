# Real-Time Face Recognition

A simple, lightweight real-time face recognition system built in Python using OpenCV and the `face_recognition` library.

## Features
- **Data Collection:** `collect.py` detects faces via webcam, crops them to eliminate background noise, and saves exactly 30 images to build a dataset.
- **Face Encoding:** `encode.py` generates 128D facial embeddings for each person in the dataset and saves them to a pickle file.
- **Real-Time Recognition:** `recognize.py` runs a live webcam feed, matches faces against the trained embeddings (with a strict < 0.45 distance threshold), and draws a green bounding box around recognized individuals. Unrecognized faces are labeled as "Unknown".

## Requirements
- Python 3.7+
- A working webcam

## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/thejas-06/Real-Time-Face-Recognition.git
cd Real-Time-Face-Recognition
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Installing `face_recognition` may require CMake and a C++ compiler depending on your OS).*

## Usage

1. **Collect Data:** Run `collect.py` to capture training images for a new person. Enter their name when prompted and face the camera. The script will automatically crop and save 30 face images.

2. **Train Model:** Run `encode.py` to process the newly collected images and update the `encodings.pkl` file with their facial embeddings.

3. **Recognize Faces:** Run `recognize.py` to start the live webcam feed and see face recognition in action. Press `Q` to quit at any time.

## Project Structure
- `collect.py`: Webcam script for dataset collection.
- `encode.py`: Script to generate facial embeddings.
- `recognize.py`: Real-time face matching application.
- `requirements.txt`: Python package dependencies.
- `dataset/`: Directory containing folders of training images (ignored by git).
- `encodings.pkl`: Serialized face embeddings (ignored by git).
