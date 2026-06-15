```markdown
# Spark Manufacturing Project: YOLO Object Detection

A high-performance, real-time object detection pipeline optimized for local machine vision and manufacturing environments. This project uses custom-trained YOLO weights to run inference on live camera feeds, static images, and pre-recorded video logs. It also includes a quantized Edge-AI deployment format for lightweight hardware like the Raspberry Pi.

## 🚀 Setup and Installation

1. Activate your virtual environment (Recommended).
2. Install the required dependencies: pip install ultralytics opencv-python numpy

```



## 🛠️ Running the Model

To run inference, use the `yolo_detect.py` script. You must specify the path to your trained model weights and your desired input source.

**Standard Execution Command:**

```bash
python yolo_detect.py --model object_detection/my_model/weights/best.pt --source usb0 --resolution 1280x720

```

### 📹 Input Sources (`--source`)

The `--source` argument dynamically maps your local hardware devices or files. Use the parameters below depending on your setup:

* `usb0` : External USB webcam stream (Default target for external hardware modules).
* `0` : Integrated system camera (Built-in MacBook FaceTime HD Camera or laptop webcam).
* `1` : Secondary system camera index (Commonly targets an iPhone via Continuity Camera on Mac).
* `picamera0` : Native Raspberry Pi camera interface.
* `path/to/video.mp4` : Runs continuous inference through a pre-recorded video clip.
* `path/to/folder/` : Iterates sequentially through a folder of testing images.
* `path/to/image.jpg` : Runs inference on a single static image.

### 🎛️ Live View Controls

When the inference display window is active, use these keyboard inputs to control the stream:

* `q` or `Q`: Quit the program and safely close all hardware streams.
* `s` or `S`: Pause the live stream frame buffer. Press any key to resume.
* `p` or `P`: Capture and save a screenshot of the current annotated frame locally as `capture.png`.

## 👏 Acknowledgments & Credits

A special thanks to **Evan Juras and EJ Technology Consultants** for their foundational work and tutorials on YOLO model training and deployment.

* Website: [https://www.ejtech.io/](https://www.ejtech.io/)
* GitHub: [EdjeElectronics](https://github.com/EdjeElectronics)
