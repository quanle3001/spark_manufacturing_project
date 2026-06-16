import csv
import json
from collections import Counter
from datetime import datetime

from config import (
    DATASET_DIR,
    MODEL_DIR,
    DATASET_INDEX_PATH,
    LABELS_JSON_PATH,
    TRAINING_INFO_PATH,
    CLASSES,
)


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


def write_dataset_index(rows):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with DATASET_INDEX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "relative_path", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return DATASET_INDEX_PATH


def write_labels_json():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    label_data = {
        "classes": CLASSES,
        "label_meaning": {
            "defect-free": "good object inside ROI",
            "defective": "bad/defective object inside ROI",
            "conveyor": "empty conveyor/background inside ROI",
        },
    }

    LABELS_JSON_PATH.write_text(json.dumps(label_data, indent=2), encoding="utf-8")
    return LABELS_JSON_PATH


def write_training_info(rows):
    counts = Counter(row["label"] for row in rows)
    generated_at = datetime.now().isoformat(timespec="seconds")

    text = f"""TensorFlow Lite Dataset / Label Index Report
Generated: {generated_at}

Important:
This TensorFlow Lite version does not train a model inside this script.
Train/export your model separately, then place the .tflite file in models/.

Expected model path:
models/roi_tflite_defect_sorter.tflite

Automatic labels are based on folder names:

dataset/defect-free/  -> defect-free
dataset/defective/    -> defective
dataset/conveyor/     -> conveyor

Class counts:
""" + "\n".join([f"{class_name}: {counts.get(class_name, 0)}" for class_name in CLASSES]) + f"""

Total images:
{len(rows)}
"""

    TRAINING_INFO_PATH.write_text(text, encoding="utf-8")
    return TRAINING_INFO_PATH


def main():
    rows = find_dataset_images()
    counts = Counter(row["label"] for row in rows)

    dataset_index = write_dataset_index(rows)
    labels_json = write_labels_json()
    training_info = write_training_info(rows)

    print("\nCreated TensorFlow Lite dataset label files:")
    print(f"  Dataset index: {dataset_index}")
    print(f"  Labels JSON:    {labels_json}")
    print(f"  Info report:    {training_info}")

    print("\nImage counts:")
    for class_name in CLASSES:
        print(f"  {class_name}: {counts.get(class_name, 0)}")

    if len(rows) == 0:
        print("\nWarning: no dataset images found yet.")

    print("\nDone.")


if __name__ == "__main__":
    main()
