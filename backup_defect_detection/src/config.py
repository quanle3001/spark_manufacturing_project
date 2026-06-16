from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"

# Real demo TensorFlow Lite model.
# Put your trained .tflite file here.
TFLITE_MODEL_PATH = MODEL_DIR / "roi_tflite_defect_sorter.tflite"
LABELS_PATH = MODEL_DIR / "tflite_labels.txt"

# Files automatically created by prepare_dataset_index.py / train_model.py.
DATASET_INDEX_PATH = MODEL_DIR / "dataset_index.csv"
LABELS_JSON_PATH = MODEL_DIR / "labels.json"
TRAINING_INFO_PATH = MODEL_DIR / "tflite_training_info.txt"

# ----------------------------
# Dataset classes
# The order must match models/tflite_labels.txt and your TFLite output order.
# ----------------------------
DEFECT_FREE_CLASS = "defect-free"
DEFECTIVE_CLASS = "defective"
CONVEYOR_CLASS = "conveyor"

CLASSES = [DEFECT_FREE_CLASS, DEFECTIVE_CLASS, CONVEYOR_CLASS]
OBJECT_CLASSES = [DEFECT_FREE_CLASS, DEFECTIVE_CLASS]

# ----------------------------
# Camera settings
# Keep 640x480 because the guide-style ROI coordinates are based on this size.
# ----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ----------------------------
# Fixed ROI settings
# roi = (x1, y1, x2, y2)
# ----------------------------
ROI = (200, 100, 440, 380)

# OpenCV uses BGR color order.
ROI_COLOR_BLUE = (255, 0, 0)

# ----------------------------
# Counter / sorting settings
# The counter waits for stable object/conveyor states to avoid double-counting.
# ----------------------------
SMOOTH_WINDOW = 7
MIN_OBJECT_VOTES_BEFORE_COUNT = 3
RESET_CONVEYOR_FRAMES = 5

# Send servo push only after the bad object is counted once.
DEFAULT_SERVO_DELAY_SECONDS = 0.0
