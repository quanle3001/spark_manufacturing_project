import argparse
import cv2

from config import FRAME_WIDTH, FRAME_HEIGHT
from roi_utils import draw_roi, parse_roi


def main():
    parser = argparse.ArgumentParser(description="Test webcam and fixed ROI frame.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--display-scale", type=float, default=1.5)
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()

    roi = parse_roi(args.roi)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try --camera-index 1 or 2.")
        return

    window_name = "Camera ROI Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(args.width * args.display_scale), int(args.height * args.display_scale))

    print("Camera opened successfully.")
    print(f"Using ROI: {roi}")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read frame.")
            break

        display = frame.copy()
        draw_roi(display, roi)

        cv2.putText(
            display,
            f"Camera {args.camera_index} | ROI {roi} | q quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
        )

        if args.display_scale != 1.0:
            display = cv2.resize(display, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow(window_name, display)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
