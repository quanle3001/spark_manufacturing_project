from collections import Counter
import argparse
import csv
import json
from datetime import datetime

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


def find_dataset_images():
    rows = []

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rows.append({
                "image_path": str(image_path),
                "relative_path": str(image_path.relative_to(DATASET_DIR.parent)),
                "label": class_name,
            })

    return rows


def save_dataset_index(rows):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    index_path = MODEL_DIR / "dataset_index.csv"

    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "relative_path", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return index_path


def save_labels_json():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = MODEL_DIR / "labels.json"

    label_data = {
        "classes": CLASSES,
        "label_meaning": {
            "defect-free": "good object inside ROI",
            "defective": "bad/defective object inside ROI",
            "conveyor": "empty conveyor/background inside ROI",
        }
    }

    labels_path.write_text(json.dumps(label_data, indent=2), encoding="utf-8")
    return labels_path


def load_dataset(rows, roi):
    X = []
    y = []

    for row in rows:
        image_path = row["image_path"]
        label = row["label"]

        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: could not read image: {image_path}")
            continue

        image = maybe_crop_full_frame_image(image, roi)
        features = extract_features(image)

        X.append(features)
        y.append(label)

    if not X:
        raise RuntimeError("No usable images found. Capture images first.")

    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(
        description="Train ROI defect detection model.")
    parser.add_argument("--roi", nargs=4, type=int, default=None)
    parser.add_argument("--min-images-per-class", type=int, default=20)
    args = parser.parse_args()

    roi = parse_roi(args.roi)

    print("\n==============================")
    print("ROI DEFECT MODEL TRAINING")
    print("==============================")
    print(f"Dataset folder: {DATASET_DIR}")
    print(f"Model output:   {MODEL_PATH}")
    print(f"ROI:            {roi}")

    rows = find_dataset_images()

    index_path = save_dataset_index(rows)
    labels_path = save_labels_json()

    print("\nAutomatic label files created:")
    print(f"Dataset index: {index_path}")
    print(f"Labels file:   {labels_path}")

    if not rows:
        raise RuntimeError(
            "\nNo images found.\n"
            "You need images inside:\n"
            "dataset/defect-free/\n"
            "dataset/defective/\n"
            "dataset/conveyor/\n"
        )

    counts = Counter(row["label"] for row in rows)

    print("\nImage counts:")
    for class_name in CLASSES:
        print(f"{class_name}: {counts.get(class_name, 0)} image(s)")

    missing_classes = [
        class_name for class_name in CLASSES
        if counts.get(class_name, 0) == 0
    ]

    if missing_classes:
        raise RuntimeError(
            "Missing images for class(es): " + ", ".join(missing_classes)
        )

    low_count_classes = [
        class_name for class_name in CLASSES
        if counts.get(class_name, 0) < args.min_images_per_class
    ]

    if low_count_classes:
        print("\nWarning: these classes have too few images:")
        for class_name in low_count_classes:
            print(f"{class_name}: {counts.get(class_name, 0)} image(s)")
        print("The model may train, but detection may be bad.")

    print("\nLoading images and extracting ROI features...")
    X, y = load_dataset(rows, roi)

    actual_counts = Counter(y)

    print(f"\nTotal usable images loaded: {len(y)}")
    print(f"Feature vector size: {X.shape[1]}")

    min_class_count = min(actual_counts.values())

    if len(y) < 12 or min_class_count < 3:
        print("\nDataset is very small.")
        print("Training and testing on the same data for now.")
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, y_pred, labels=CLASSES)

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(matrix)

    print("Labels order:")
    print(CLASSES)

    trained_at = datetime.now().isoformat(timespec="seconds")

    payload = {
        "model": model,
        "classes": CLASSES,
        "roi": roi,
        "feature_version": "roi_hsv_hist_gray_edge_v2",
        "trained_at": trained_at,
        "dataset_counts": dict(actual_counts),
        "total_images": int(len(y)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH)

    report_path = MODEL_DIR / "training_report.txt"

    report_text = f"""ROI Defect Detection Training Report
Generated: {trained_at}

Dataset folder:
{DATASET_DIR}

Model path:
{MODEL_PATH}

ROI:
{roi}

Class counts:
""" + "\n".join(
        [f"{class_name}: {actual_counts.get(class_name, 0)}" for class_name in CLASSES]
    ) + f"""

Total usable images:
{len(y)}

Feature vector size:
{X.shape[1]}

Classification report:
{report}

Confusion matrix:
{matrix}

Labels order:
{CLASSES}
"""

    report_path.write_text(report_text, encoding="utf-8")

    print("\nSaved files:")
    print(f"Model:           {MODEL_PATH}")
    print(f"Labels:          {labels_path}")
    print(f"Dataset index:   {index_path}")
    print(f"Training report: {report_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
