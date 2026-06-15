from collections import Counter
import argparse

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import DATASET_DIR, MODEL_DIR, MODEL_PATH, CLASSES
from features import extract_features
from roi_utils import maybe_crop_full_frame_image, parse_roi


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_dataset(roi):
    X = []
    y = []

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            print(f"Missing folder: {class_dir}")
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: could not read {image_path}")
                continue

            image = maybe_crop_full_frame_image(image, roi)
            features = extract_features(image)

            X.append(features)
            y.append(class_name)

    if not X:
        raise RuntimeError("No images found. Capture images first using capture_roi_images.py.")

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train the ROI defect detection model.")
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()

    roi = parse_roi(args.roi)

    print(f"Loading dataset from: {DATASET_DIR}")
    print(f"Using ROI for full-frame fallback crop: {roi}")

    X, y = load_dataset(roi)

    counts = Counter(y)
    print("\nClass counts:")
    for class_name in CLASSES:
        print(f"  {class_name}: {counts.get(class_name, 0)}")

    if len(counts) < 2:
        raise RuntimeError("Need at least two classes with images to train a useful model.")

    min_class_count = min(counts.values())
    if min_class_count < 5:
        print("\nWarning: very small dataset. Capture more images for better results.")

    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))
    print("Labels order:", CLASSES)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "classes": CLASSES,
        "roi": roi,
        "feature_version": "roi_hsv_hist_gray_edge_v1",
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
