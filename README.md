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

While the application is running, press `q` to quit the application.

Command-line options are available to change the camera index, the assumed face
width, and the horizontal field of view used for the distance estimate:

```bash
python face_distance.py --camera-index 0 --known-face-width 16.0 --horizontal-fov 60.0
```

Adjust the numbers based on your camera setup and your own measurements for more
accurate results. The distance estimate is calculated automatically using the
provided field-of-view value.
