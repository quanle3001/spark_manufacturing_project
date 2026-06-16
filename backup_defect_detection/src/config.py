from pathlib import Path

# ----------------------------
# Project paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "defect_model.joblib"

# ----------------------------
# Dataset classes
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
# These values match the guide-style blue box:
# roi = (x1, y1, x2, y2)
# ----------------------------
ROI = (200, 100, 440, 380)

# OpenCV uses BGR color order.
ROI_COLOR_BLUE = (255, 0, 0)

# ----------------------------
# Image preprocessing settings
# The ROI crop is resized to this shape before feature extraction.
# ----------------------------
IMAGE_SIZE = (64, 64)  # width, height

# ----------------------------
# Counter settings
# The counter waits for stable object/conveyor states to avoid double-counting.
# ----------------------------
SMOOTH_WINDOW = 7
MIN_OBJECT_VOTES_BEFORE_COUNT = 3
RESET_CONVEYOR_FRAMES = 5
