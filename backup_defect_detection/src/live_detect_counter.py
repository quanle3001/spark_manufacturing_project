import argparse
from collections import Counter, deque
import time

import cv2

from config import (
    TFLITE_MODEL_PATH,
    LABELS_PATH,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ROI,
    DEFECT_FREE_CLASS,
    DEFECTIVE_CLASS,
    CONVEYOR_CLASS,
    OBJECT_CLASSES,
    SMOOTH_WINDOW,
    MIN_OBJECT_VOTES_BEFORE_COUNT,
    RESET_CONVEYOR_FRAMES,
    DEFAULT_SERVO_DELAY_SECONDS,
)
from roi_utils import crop_roi, draw_roi, parse_roi
from tflite_utils import (
    load_tflite_interpreter,
    load_labels,
    describe_tflite_model,
    predict_classification,
)
from arduino_sorter import ArduinoSorter


def majority_vote(values):
    if not values:
        return None

    return Counter(values).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser(description="Live TensorFlow Lite ROI detection with counter and optional Arduino servo sorting.")
    parser.add_argument("--model", default=str(TFLITE_MODEL_PATH))
    parser.add_argument("--labels", default=str(LABELS_PATH))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--display-scale", type=float, default=1.5)
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))

    parser.add_argument("--arduino-enabled", action="store_true")
    parser.add_argument("--arduino-port", default="/dev/ttyACM0")
    parser.add_argument("--arduino-baudrate", type=int, default=9600)
    parser.add_argument("--servo-delay", type=float, default=DEFAULT_SERVO_DELAY_SECONDS)
    args = parser.parse_args()

    roi = parse_roi(args.roi if args.roi is not None else ROI)

    labels = load_labels(args.labels)

    interpreter = load_tflite_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details, output_details = describe_tflite_model(interpreter)

    sorter = ArduinoSorter(
        port=args.arduino_port,
        baudrate=args.arduino_baudrate,
        enabled=args.arduino_enabled,
        cooldown_seconds=1.0,
    )
    sorter.connect()

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try --camera-index 1 or 2.")
        sorter.close()
        return

    window_name = "TFLite ROI Defect Detection Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(args.width * args.display_scale), int(args.height * args.display_scale))

    recent_predictions = deque(maxlen=SMOOTH_WINDOW)

    good_count = 0
    defective_count = 0

    object_active = False
    object_votes = []
    conveyor_streak = 0

    print("\nLive TFLite ROI detection started.")
    print(f"Using ROI: {roi}")
    print(f"Arduino enabled: {args.arduino_enabled}")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read frame.")
            break

        roi_crop = crop_roi(frame, roi)

        label, confidence, _scores = predict_classification(
            interpreter,
            input_details,
            output_details,
            roi_crop,
            labels,
        )

        recent_predictions.append(label)
        smooth_label = majority_vote(recent_predictions)

        if smooth_label in OBJECT_CLASSES:
            object_active = True
            conveyor_streak = 0
            object_votes.append(smooth_label)

        elif smooth_label == CONVEYOR_CLASS:
            conveyor_streak += 1

            # Count only after the object leaves the ROI and conveyor/background is stable again.
            if object_active and conveyor_streak >= RESET_CONVEYOR_FRAMES:
                if len(object_votes) >= MIN_OBJECT_VOTES_BEFORE_COUNT:
                    final_label = majority_vote(object_votes)

                    if final_label == DEFECT_FREE_CLASS:
                        good_count += 1
                        print(f"[Count] Good object counted. Total good: {good_count}")

                    elif final_label == DEFECTIVE_CLASS:
                        defective_count += 1
                        print(f"[Count] Defective object counted. Total defective: {defective_count}")

                        if args.servo_delay > 0:
                            print(f"[Servo] Waiting {args.servo_delay} second(s) before push.")
                            time.sleep(args.servo_delay)

                        sorter.push_bad()

                object_active = False
                object_votes = []

        display = frame.copy()

        if smooth_label == DEFECTIVE_CLASS:
            color = (0, 0, 255)
            status = f"Defective ({confidence:.2f})"
        elif smooth_label == DEFECT_FREE_CLASS:
            color = (0, 255, 0)
            status = f"Good Quality ({confidence:.2f})"
        else:
            color = (255, 0, 0)
            status = f"Conveyor ({confidence:.2f})"

        draw_roi(display, roi, color=color, thickness=2)

        cv2.putText(display, f"Good Quality: {good_count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(display, f"Defective: {defective_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(display, f"Prediction: {status}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        cv2.putText(display, "q quit", (20, display.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if args.display_scale != 1.0:
            display = cv2.resize(display, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow(window_name, display)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    sorter.close()

    print("\nFinal counts:")
    print(f"Good Quality: {good_count}")
    print(f"Defective: {defective_count}")


if __name__ == "__main__":
    main()
