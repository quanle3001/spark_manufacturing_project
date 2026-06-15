import argparse

import cv2
import joblib

from config import MODEL_PATH
from features import extract_features
from roi_utils import maybe_crop_full_frame_image, parse_roi


def main():
    parser = argparse.ArgumentParser(description="Predict one image using the trained ROI model.")
    parser.add_argument("image_path", help="Path to an image file.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to trained model.")
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()

    payload = joblib.load(args.model_path)
    model = payload["model"]

    roi = parse_roi(args.roi if args.roi is not None else payload.get("roi"))

    image = cv2.imread(args.image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image_path}")

    image = maybe_crop_full_frame_image(image, roi)

    features = extract_features(image).reshape(1, -1)
    prediction = model.predict(features)[0]

    confidence_text = ""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        best = probs.max()
        confidence_text = f" | confidence: {best:.2f}"

    print(f"Prediction: {prediction}{confidence_text}")


if __name__ == "__main__":
    main()
