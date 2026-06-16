import argparse
import cv2

from config import TFLITE_MODEL_PATH, LABELS_PATH, ROI
from roi_utils import crop_roi, parse_roi
from tflite_utils import (
    load_tflite_interpreter,
    load_labels,
    describe_tflite_model,
    predict_classification,
)


def main():
    parser = argparse.ArgumentParser(description="Predict one image using TensorFlow Lite.")
    parser.add_argument("image_path")
    parser.add_argument("--model", default=str(TFLITE_MODEL_PATH))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()

    roi = parse_roi(args.roi if args.roi is not None else ROI)

    labels = load_labels(args.labels)

    interpreter = load_tflite_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details, output_details = describe_tflite_model(interpreter)

    image = cv2.imread(args.image_path)

    if image is None:
        raise RuntimeError(f"Could not read image: {args.image_path}")

    # If image is a full 640x480 camera frame, crop ROI.
    # If it is already an ROI crop and smaller than the ROI dimensions, use as-is.
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi

    if w >= x2 and h >= y2:
        image_for_prediction = crop_roi(image, roi)
    else:
        image_for_prediction = image

    label, confidence, scores = predict_classification(
        interpreter,
        input_details,
        output_details,
        image_for_prediction,
        labels,
    )

    print(f"Prediction: {label}")
    print(f"Confidence/score: {confidence:.4f}")
    print("Scores:")
    for name, score in zip(labels, scores):
        print(f"  {name}: {float(score):.4f}")


if __name__ == "__main__":
    main()
