# Face Distance Estimator

This repository contains a Python script that tracks faces with your laptop
camera, places a magenta crosshair on the tip of each detected nose, draws a
bounding box around every face, estimates how far they are from the camera in
centimeters, and adds high-confidence gender and age predictions underneath the
box using DeepFace's state-of-the-art models.

## Quick start (Windows, macOS, or Linux)

1. Download the project files once (for example with **Code → Download ZIP** on
   GitHub) or clone it with Git. Extract the ZIP file and keep the resulting
   `hello-world` folder anywhere you like (e.g., Documents). You do **not** need
   to copy the files again later—just re-use this folder every time you want to
   run the app.
2. Make sure you have Python 3.8 or later installed. On Windows you can verify
   by opening **PowerShell** and running `python --version` or `py --version`.
3. Open a terminal in the project folder:
   - **Windows:** Right-click inside the folder and choose “Open in Terminal”,
     or open PowerShell and run `cd C:\path\to\hello-world`.
   - **macOS/Linux:** Open Terminal and run `cd /path/to/hello-world`.
4. Run the convenience launcher. It installs the required packages (including
   DeepFace for age and gender recognition) the first time and then starts the
   webcam app automatically:

   ```bash
   python run_face_distance.py
   ```

   On Windows, if `python` is not found, try `py run_face_distance.py` instead.
5. Grant camera access if your system asks for permission. When the window
   opens, you should see green boxes with the red “Hedef” label, a magenta
   crosshair centered on each nose with a small gap in the middle, a blue
   distance readout in centimeters hugging the lower-right side of every box,
   and a yellow line underneath the box showing the predicted gender and age.
   Press `q` to close the window when you are done.

   If the yellow age/gender line does not appear, check the terminal once:
   the app prints a one-time warning if DeepFace or TensorFlow is missing so
   you know which dependency to install.

> **Note:** The first run may take a little longer because DeepFace downloads
> pretrained weights for its high-accuracy models. Subsequent runs will start
> immediately and reuse the cached weights.

### Running the script directly

If you prefer to manage the dependencies yourself, install them once with pip:

```bash
pip install opencv-python mediapipe numpy deepface
```

Then run the main script:

```bash
python face_distance.py
```

### Optional command-line arguments

You can adjust the camera index, the assumed face width, and the horizontal
field of view used for the distance estimate:

```bash
python face_distance.py --camera-index 0 --known-face-width 16.0 --horizontal-fov 60.0
```

Adjust the numbers based on your camera setup and your own measurements for more
accurate results. The distance estimate is calculated automatically using the
provided field-of-view value.
