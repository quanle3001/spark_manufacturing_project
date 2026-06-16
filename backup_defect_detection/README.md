# TensorFlow Lite ROI Defect Detection + Servo Sorting

This is the refactored TensorFlow Lite version of the previous ROI defect-detection project.

It keeps the previous features:

```text
Fixed blue ROI frame
ROI-only image capture
Dataset counts while capturing
Automatic labels from folder names
Bigger camera preview window
Live good/defective counter
Conveyor/background reset logic to avoid double-counting
Optional Arduino + servo sorting
```

The main change is:

```text
Old version:
Random Forest model -> models/defect_model.joblib

New version:
TensorFlow Lite model -> models/roi_tflite_defect_sorter.tflite
```

---

## Expected model files

Put your real TensorFlow Lite model here:

```text
models/roi_tflite_defect_sorter.tflite
```

The labels file is:

```text
models/tflite_labels.txt
```

Default label order:

```text
defect-free
defective
conveyor
```

The order in `tflite_labels.txt` must match the output order of your TFLite model.

---

## Important model requirement

This version expects a **classification-style TFLite model**.

Expected behavior:

```text
Input: one ROI image
Output: 3 class scores/probabilities
```

The 3 classes should be:

```text
defect-free
defective
conveyor
```

If your `.tflite` model is YOLO/object-detection style, then only this file should need a different post-processing function:

```text
src/tflite_utils.py
```

Run the validator first to check.

---

# Mac testing

## 1. Install packages

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Mac, you may also need TensorFlow if `tflite_runtime` is not available:

```bash
pip install tensorflow
```

## 2. Test camera and ROI

```bash
python src/test_camera_roi.py --camera-index 0 --display-scale 1.5
```

If needed:

```bash
python src/test_camera_roi.py --camera-index 1 --display-scale 1.5
```

## 3. Capture ROI dataset images

```bash
python src/capture_roi_images.py --class-name defect-free --camera-index 1 --display-scale 1.5
python src/capture_roi_images.py --class-name defective --camera-index 1 --display-scale 1.5
python src/capture_roi_images.py --class-name conveyor --camera-index 1 --display-scale 1.5
```

Controls:

```text
c or SPACE = capture ROI image
q          = quit
```

Images are saved with names like:

```text
defect-free_000001.jpg
defective_000001.jpg
conveyor_000001.jpg
```

## 4. Create automatic dataset label files

This replaces the old Random Forest training step.

```bash
python src/train_model.py
```

It creates:

```text
models/dataset_index.csv
models/labels.json
models/tflite_training_info.txt
```

It does not train the TFLite model. Your TFLite model should already be exported separately.

## 5. Validate the TFLite model

After placing your `.tflite` model in `models/`:

```bash
python src/validate_tflite_model.py --model models/roi_tflite_defect_sorter.tflite
```

If output size matches the label count, the live classifier should work.

## 6. Test one image

```bash
python src/predict_image.py dataset/defective/defective_000001.jpg --model models/roi_tflite_defect_sorter.tflite
```

## 7. Run live detection without Arduino

```bash
python src/live_detect_counter.py --model models/roi_tflite_defect_sorter.tflite --camera-index 1 --display-scale 1.5
```

---

# Raspberry Pi setup

## 1. Install system packages

```bash
sudo apt update
sudo apt install python3-pip python3-opencv -y
```

## 2. Install Python packages

```bash
pip3 install numpy pyserial
```

Install TensorFlow Lite runtime:

```bash
pip3 install tflite-runtime
```

If that does not work on your Pi OS, use the TensorFlow package you already used for your TFLite demo.

## 3. Test camera and ROI

```bash
python3 src/test_camera_roi.py --camera-index 0 --display-scale 1.5
```

## 4. Validate model

```bash
python3 src/validate_tflite_model.py --model models/roi_tflite_defect_sorter.tflite
```

## 5. Run live detection without Arduino first

```bash
python3 src/live_detect_counter.py --model models/roi_tflite_defect_sorter.tflite --camera-index 0 --display-scale 1.5
```

---

# Arduino + servo sorting

## Wiring

```text
Raspberry Pi USB port -> Arduino USB cable

Servo signal wire  -> Arduino pin 9
Servo red wire     -> external 5V power
Servo brown/black  -> external GND
Arduino GND        -> external power GND
```

Do not power a medium/large servo directly from the Raspberry Pi. Use external 5V servo power and connect all grounds together.

## Upload Arduino code

Open and upload this file in Arduino IDE:

```text
arduino/servo_sorter/servo_sorter.ino
```

## Test Arduino servo manually

Without actually sending commands:

```bash
python3 src/test_arduino_servo.py --port /dev/ttyACM0
```

Actually connect and test:

```bash
python3 src/test_arduino_servo.py --port /dev/ttyACM0 --enabled
```

If your Arduino appears as USB serial:

```bash
python3 src/test_arduino_servo.py --port /dev/ttyUSB0 --enabled
```

## Run live detection with Arduino enabled

```bash
python3 src/live_detect_counter.py \
  --model models/roi_tflite_defect_sorter.tflite \
  --camera-index 0 \
  --display-scale 1.5 \
  --arduino-enabled \
  --arduino-port /dev/ttyACM0
```

If the servo is physically after the camera ROI, add a delay:

```bash
python3 src/live_detect_counter.py \
  --model models/roi_tflite_defect_sorter.tflite \
  --camera-index 0 \
  --display-scale 1.5 \
  --arduino-enabled \
  --arduino-port /dev/ttyACM0 \
  --servo-delay 0.5
```

---

# How sorting logic works

The code does not send `BAD` every frame.

It waits for this sequence:

```text
1. ROI is conveyor/background.
2. Object enters ROI.
3. TFLite model predicts defect-free or defective.
4. Object leaves ROI.
5. ROI becomes conveyor/background again.
6. If the final object label was defective, send BAD once to Arduino.
7. Servo pushes once and returns.
```

This prevents the same object from being counted or pushed repeatedly.

---

# Main files

```text
src/live_detect_counter.py       real live TFLite detection + counter + optional servo
src/tflite_utils.py              TFLite input/output handling
src/capture_roi_images.py        capture ROI dataset images
src/train_model.py               creates label/index files, no Random Forest training
src/validate_tflite_model.py     checks model shape and labels
src/predict_image.py             test one image
src/arduino_sorter.py            Raspberry Pi serial helper
src/test_arduino_servo.py        manual servo test
arduino/servo_sorter/servo_sorter.ino
```