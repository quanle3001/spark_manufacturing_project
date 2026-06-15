import cv2
import numpy as np

from config import IMAGE_SIZE


def preprocess_image(image):
    """Resize an ROI crop to a consistent size."""
    if image is None:
        raise ValueError("Image is None. Check the image path or camera frame.")

    resized = cv2.resize(image, IMAGE_SIZE)
    return resized


def extract_features(image):
    """
    Extract lightweight classical ML features from the ROI crop.

    Features:
    - HSV color histogram
    - grayscale mean/std
    - edge density

    This is intentionally lightweight so it can run on Raspberry Pi.
    """
    img = preprocess_image(image)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.array([gray.mean()], dtype=np.float32)
    std = np.array([gray.std()], dtype=np.float32)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([np.count_nonzero(edges) / edges.size], dtype=np.float32)

    features = np.concatenate([hist, mean, std, edge_density]).astype(np.float32)
    return features
