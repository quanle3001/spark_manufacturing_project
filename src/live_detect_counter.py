import argparse
from collections import Counter, deque

import cv2
import joblib
import numpy as np

from config import (
    MODEL_PATH,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DEFECT_FREE_CLASS,
    DEFECTIVE_CLASS,
    CONVEYOR_CLASS,
    OBJECT_CLASSES,
    SMOOTH_WINDOW,
    MIN_OBJECT_VOTES_BEFORE_COUNT,
    RESET_CONVEYOR_FRAMES,
)
from features import extract_features
from roi_utils import crop_roi, draw_roi, parse_roi


def majority_vote(values):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser(description="Run live ROI defect detection with good/bad counter.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index. Try 0, 1, or 2.")
    parser.add_argument("--model-path", default=str(MODEL_PATH), help="Path to trained model.")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.5,
        help="Scale factor for the preview window. Example: 1.5 or 2.0.",
    )
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--reset-conveyor-frames", type=int, default=RESET_CONVEYOR_FRAMES)
    args = parser.parse_args()

    payload = joblib.load(args.model_path)
    model = payload["model"]

    roi = parse_roi(args.roi if args.roi is not None else payload.get("roi"))

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try --camera-index 1 or 2.")
        return

    recent_predictions = deque(maxlen=args.smooth_window)

    good_count = 0
    defective_count = 0

    object_active = False
    object_votes = []
    conveyor_streak = 0

    window_name = "ROI Defect Detection Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(args.width * args.display_scale), int(args.height * args.display_scale))

    print("Live ROI defect detection started.")
    print(f"Using ROI: {roi}")
    print(f"Display scale: {args.display_scale}")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame.")
            break

        roi_crop = crop_roi(frame, roi)
        features = extract_features(roi_crop).reshape(1, -1)

        raw_pred = model.predict(features)[0]
        recent_predictions.append(raw_pred)
        smooth_pred = majority_vote(recent_predictions)

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = float(np.max(probs))

        # Track object presence inside the ROI.
        # "conveyor" means empty/background. Objects are defect-free or defective.
        if smooth_pred in OBJECT_CLASSES:
            object_active = True
            conveyor_streak = 0
            object_votes.append(smooth_pred)
        elif smooth_pred == CONVEYOR_CLASS:
            conveyor_streak += 1

            # Count the object only after it leaves the ROI and conveyor is stable again.
            if object_active and conveyor_streak >= args.reset_conveyor_frames:
                if len(object_votes) >= MIN_OBJECT_VOTES_BEFORE_COUNT:
                    final_object_label = majority_vote(object_votes)
                    if final_object_label == DEFECT_FREE_CLASS:
                        good_count += 1
                    elif final_object_label == DEFECTIVE_CLASS:
                        defective_count += 1

                object_active = False
                object_votes = []

        display = frame.copy()

        if smooth_pred == DEFECTIVE_CLASS:
            box_color = (0, 0, 255)
            status_text = "Prediction: Defective"
        elif smooth_pred == DEFECT_FREE_CLASS:
            box_color = (0, 255, 0)
            status_text = "Prediction: Good Quality"
        else:
            box_color = (255, 0, 0)
            status_text = "Prediction: Conveyor"

        draw_roi(display, roi, color=box_color, thickness=2)

        if confidence is not None:
            status_text += f" ({confidence:.2f})"

        cv2.putText(
            display,
            f"Good Quality: {good_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"Defective: {defective_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            display,
            status_text,
            (roi[0], max(roi[1] - 12, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
        )

        cv2.putText(
            display,
            "q quit",
            (20, display.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if args.display_scale != 1.0:
            display = cv2.resize(display, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal counts:")
    print(f"Good Quality: {good_count}")
    print(f"Defective: {defective_count}")


if __name__ == "__main__":
    main()
