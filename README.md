# AI Defect Detection and Sorting System

## Project Overview

This project is an AI-powered defect detection and sorting system designed to identify faulty products on an assembly line. The goal of the project is to explore how artificial intelligence, computer vision, and embedded systems can work together to improve manufacturing quality control.

Traditional defect inspection often depends on human workers visually checking products, which can be slow, inconsistent, and affected by fatigue. This project demonstrates how an AI system can analyze camera images in real time, classify products as either good or defective, and support automated sorting using hardware such as a Raspberry Pi, Arduino, and servo motor.

## Purpose

The purpose of this project is to build a small-scale manufacturing inspection system that can:

* Capture images from a camera using a fixed Region of Interest (ROI)
* Detect whether an object is defect-free, defective, or just background/conveyor
* Count good and defective objects as they pass through the ROI
* Use a TensorFlow Lite model for real-time defect classification
* Send a signal to an Arduino-controlled servo motor to push defective items out of the line
* Demonstrate how AI can improve speed, consistency, and reliability in quality control

## How It Works

The system uses a camera to monitor a fixed frame where objects pass through. This fixed frame is called the Region of Interest, or ROI. Instead of analyzing the entire camera image, the program focuses only on the ROI, which helps make detection more consistent and efficient.

The workflow is:

```text
Camera captures image
        ↓
Program crops the fixed ROI
        ↓
TensorFlow Lite model classifies the ROI
        ↓
Object is labeled as defect-free, defective, or conveyor/background
        ↓
System counts good and defective objects
        ↓
If defective, Raspberry Pi sends command to Arduino
        ↓
Arduino moves servo motor to push defective object out
```

## Dataset Collection

The project uses three dataset classes:

```text
dataset/
├── defect-free/
├── defective/
└── conveyor/
```

Images are automatically labeled based on the folder they are saved into. This means each image does not need to be manually labeled with a separate labeling tool.

For example:

```text
dataset/defect-free/  → label: defect-free
dataset/defective/    → label: defective
dataset/conveyor/     → label: conveyor
```

During image capture, the program saves only the cropped ROI area. This helps the model learn from the important part of the image instead of being distracted by the full camera background.

## Model

The main version of this project uses a TensorFlow Lite model for real-time inference on Raspberry Pi.

Expected model file:

```text
models/roi_tflite_defect_sorter.tflite
```

Expected labels file:

```text
models/tflite_labels.txt
```

The labels should be ordered as:

```text
defect-free
defective
conveyor
```

TensorFlow Lite is used because it is lightweight and suitable for edge devices such as Raspberry Pi. This allows the AI model to run locally without needing cloud processing.

## Hardware Used

The project is designed for:

* Raspberry Pi
* External USB webcam or compatible camera
* Arduino
* Servo motor
* External 5V power supply for servo
* Conveyor or small moving platform setup

The Raspberry Pi handles the camera and AI detection. The Arduino controls the servo motor used for sorting.

## Servo Sorting

When the AI model detects a defective object, the Raspberry Pi sends a command to the Arduino through USB serial communication. The Arduino then moves the servo motor to push the defective object out of the path.

The servo does not move every frame. The program waits until an object enters the ROI, gets classified, and leaves the ROI before counting it. This prevents the same object from being counted or pushed multiple times.

## Project Goal

This project demonstrates how AI can be used in modern manufacturing systems to improve product inspection. By combining computer vision, machine learning, embedded systems, and mechanical sorting, the project shows a simplified version of how automated quality control can work in real-world assembly lines.

The system is intended as an educational prototype for learning about:

* Artificial intelligence in manufacturing
* Computer vision
* TensorFlow Lite deployment
* Raspberry Pi development
* Arduino serial communication
* Servo-based sorting mechanisms
* Hardware and software integration

## Future Improvements

Possible future improvements include:

* Training the model with more images under different lighting conditions
* Improving the physical conveyor and sorting mechanism
* Adding a better object-tracking system
* Using a more advanced deep learning model
* Adding a display dashboard for live statistics
* Saving inspection results to a file or database
* Improving timing between detection and servo movement

## Memers

* Ben Vo
* Kaleem Sarpas
* Quan Le
