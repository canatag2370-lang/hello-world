# Face Distance Estimator

This repository contains a Python script that tracks a face with your laptop
camera, highlights key facial landmarks such as the nose and cheekbones, draws a
bounding box around the face, and estimates how far you are from the camera in
centimeters.

## Requirements

Install the dependencies with pip:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the script and grant access to your webcam:

```bash
python face_distance.py
```

While the application is running:

* Position your face at a known distance from the camera and press `c` to
  calibrate the distance calculation.
* Press `q` to quit the application.

Command-line options are available to change the camera index, face width used
for distance estimation, and the calibration distance:

```bash
python face_distance.py --camera-index 0 --known-face-width 16.0 --calibration-distance 50.0
```

Adjust the numbers based on your camera setup and your own measurements for more
accurate results.
