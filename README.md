# Defect Detection ROI Counter

This is a Mac-first and Raspberry-Pi-ready version of the Cytron-style defect detection project.

This version uses a **fixed Region of Interest (ROI)** like the blue frame in the guide/screenshots.

Important idea:

```text
Camera full frame
        ↓
Fixed blue ROI box only
        ↓
Crop ROI
        ↓
Train / predict using only the ROI crop
        ↓
Classify as defect-free, defective, or conveyor
        ↓
Count good and defective objects when they pass through the ROI
```

The project uses a USB/external webcam through OpenCV, so the same camera code can work on both MacBook Pro and Raspberry Pi.

---

## Project structure

```text
defect-detection-roi-counter/
│
├── dataset/
│   ├── defect-free/
│   ├── defective/
│   └── conveyor/
│
├── models/
│
├── src/
│   ├── config.py
│   ├── roi_utils.py
│   ├── features.py
│   ├── test_camera_roi.py
│   ├── capture_roi_images.py
│   ├── train_model.py
│   ├── predict_image.py
│   └── live_detect_counter.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Part A — Test and use on MacBook Pro first

## 1. Open the project

```bash
cd defect-detection-roi-counter
```

## 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install libraries

```bash
pip install -r requirements.txt
```

## 4. Test the external webcam and ROI frame

Try camera index `0` first:

```bash
python src/test_camera_roi.py --camera-index 0
```

If that opens the built-in Mac camera instead of the external webcam, try:

```bash
python src/test_camera_roi.py --camera-index 1
```

or:

```bash
python src/test_camera_roi.py --camera-index 2
```

Press `q` to quit.

The default ROI is:

```text
x1 = 200
y1 = 100
x2 = 440
y2 = 380
```

This is similar to the guide screenshot.

If the blue box is not positioned correctly for your webcam, test another ROI like this:

```bash
python src/test_camera_roi.py --camera-index 1 --roi 180 90 460 390
```

Use the same ROI values for image capture, training, and live detection.

---

## 5. Capture Mac test images

The labels are automatic because the image is saved into one of these folders:

```text
dataset/defect-free/
dataset/defective/
dataset/conveyor/
```

Capture `defect-free` images:

```bash
python src/capture_roi_images.py --class-name defect-free --camera-index 1
```

Capture `defective` images:

```bash
python src/capture_roi_images.py --class-name defective --camera-index 1
```

Capture `conveyor` / empty-background images:

```bash
python src/capture_roi_images.py --class-name conveyor --camera-index 1
```

Controls:

```text
c or SPACE = capture one ROI image
q          = quit
```

The capture window now shows your dataset balance while you are collecting images:

```text
defect-free: current image count
defective: current image count
conveyor: current image count
next filename to be saved
```

Each saved image uses a sequential filename, for example:

```text
defect-free_000001.jpg
defect-free_000002.jpg
defective_000001.jpg
conveyor_000001.jpg
```

This makes it easier to keep the three classes balanced before training.

This script displays the full camera frame with a blue ROI box, but it saves only the cropped ROI image into the dataset folder. That means the model trains only on the inside of the fixed frame.

For Mac testing, you can start small:

```text
20–50 defect-free images
20–50 defective images
20–50 conveyor images
```

Later on Raspberry Pi, collect a better dataset from the real setup.

---

## 6. Train the model on Mac

```bash
python src/train_model.py
```

This automatically:
1. Reads images from `dataset/defect-free`, `dataset/defective`, and `dataset/conveyor`.
2. Uses the folder name as the label.
3. Extracts features from each ROI crop.
4. Trains a Random Forest model.
5. Saves the model to:

```text
models/defect_model.joblib
```

---

## 7. Test one saved image

Example:

```bash
python src/predict_image.py dataset/defective/image_0001.jpg
```

---

## 8. Run live defect detection and counter on Mac

```bash
python src/live_detect_counter.py --camera-index 1
```

The live window will show:

```text
Prediction
Good count
Defective count
Fixed blue ROI box
```

The counter is designed to count one object when:
1. The ROI changes from conveyor/background to object.
2. The object is classified as defect-free or defective.
3. The ROI returns to conveyor/background.

This prevents the same object from being counted repeatedly while it stays inside the ROI.

---

# Part B — Transfer and use on Raspberry Pi later

This assumes your project repo is already uploaded to GitHub.

## 1. Clone your repo on Raspberry Pi

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

## 2. Install Raspberry Pi dependencies

```bash
sudo apt update
sudo apt install python3-pip python3-opencv libgtk-3-dev -y
pip3 install numpy scikit-learn matplotlib joblib
```

## 3. Test the USB webcam and ROI frame on Raspberry Pi

```bash
python3 src/test_camera_roi.py --camera-index 0
```

If needed:

```bash
python3 src/test_camera_roi.py --camera-index 1
python3 src/test_camera_roi.py --camera-index 2
```

If the ROI is not correct, test a different ROI:

```bash
python3 src/test_camera_roi.py --camera-index 0 --roi 200 100 440 380
```

Use the same ROI values when capturing and detecting.

---

## 4. Capture a new Raspberry Pi dataset

You should retrain using Raspberry Pi images because the camera, lighting, object position, and final setup may be different from your Mac setup.

Capture:

```bash
python3 src/capture_roi_images.py --class-name defect-free --camera-index 0
python3 src/capture_roi_images.py --class-name defective --camera-index 0
python3 src/capture_roi_images.py --class-name conveyor --camera-index 0
```

For a stronger final model, aim for:

```text
100–200 defect-free images
100–200 defective images
100–200 conveyor images
```

Keep the number of images in each class roughly balanced.

---

## 5. Retrain on Raspberry Pi

```bash
python3 src/train_model.py
```

This creates:

```text
models/defect_model.joblib
```

---

## 6. Run live detection and counting on Raspberry Pi

```bash
python3 src/live_detect_counter.py --camera-index 0
```

If your webcam is index `1`, run:

```bash
python3 src/live_detect_counter.py --camera-index 1
```

---

---

## Bigger camera window

The camera preview scripts support a larger display window with `--display-scale`.

Example for Mac:

```bash
python src/test_camera_roi.py --camera-index 1 --display-scale 2.0
python src/capture_roi_images.py --class-name defect-free --camera-index 1 --display-scale 2.0
python src/live_detect_counter.py --camera-index 1 --display-scale 2.0
```

Example for Raspberry Pi:

```bash
python3 src/test_camera_roi.py --camera-index 0 --display-scale 1.5
python3 src/capture_roi_images.py --class-name defect-free --camera-index 0 --display-scale 1.5
python3 src/live_detect_counter.py --camera-index 0 --display-scale 1.5
```

Good values to try:

```text
1.0 = original 640x480 preview
1.5 = larger preview
2.0 = much larger preview
```

This only changes the preview size. It does not change the actual camera frame, ROI coordinates, saved images, training, or detection behavior.

# Recommended workflow

```text
MacBook Pro:
1. Test external webcam
2. Adjust/check ROI
3. Capture small test dataset
4. Train model
5. Run live detection/counter

Raspberry Pi:
1. Clone repo
2. Test USB webcam
3. Adjust/check ROI
4. Capture real dataset from the Raspberry Pi setup
5. Retrain model
6. Run final live detection/counter
```

The Mac version is mainly for testing the code pipeline. The Raspberry Pi dataset should be used for the final model.
