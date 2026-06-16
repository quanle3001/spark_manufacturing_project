import argparse
from pathlib import Path

import cv2

from config import DATASET_DIR, CLASSES, FRAME_WIDTH, FRAME_HEIGHT
from roi_utils import crop_roi, draw_roi, parse_roi


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(class_name):
    """Count how many dataset images currently exist for one class."""
    class_dir = DATASET_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    return len([p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def get_class_counts():
    """Return image counts for all dataset classes."""
    return {class_name: count_images(class_name) for class_name in CLASSES}


def get_next_filename(save_dir, class_name):
    """
    Create the next sequential filename based on the number of images already saved.

    Example:
    defect-free_000001.jpg
    defect-free_000002.jpg

    This makes it easy to see how many images you have captured.
    """
    existing_count = len([p for p in save_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])

    while True:
        next_number = existing_count + 1
        filename = save_dir / f"{class_name}_{next_number:06d}.jpg"

        if not filename.exists():
            return filename, next_number

        existing_count += 1


def draw_counts_panel(frame, counts, current_class, next_number):
    """Draw dataset balance information on the camera preview."""
    x = 20
    y = 35
    line_gap = 28

    cv2.putText(
        frame,
        f"Current class: {current_class} | Next: {current_class}_{next_number:06d}.jpg",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    y += line_gap
    cv2.putText(
        frame,
        "Dataset counts:",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    for class_name in CLASSES:
        y += line_gap
        color = (255, 255, 255)

        if class_name == "defect-free":
            color = (0, 255, 0)
        elif class_name == "defective":
            color = (0, 0, 255)
        elif class_name == "conveyor":
            color = (255, 0, 0)

        prefix = "-> " if class_name == current_class else "   "
        cv2.putText(
            frame,
            f"{prefix}{class_name}: {counts.get(class_name, 0)}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )

    y += line_gap
    cv2.putText(
        frame,
        "c/SPACE capture | q quit",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    return frame


def print_counts(counts):
    print("\nCurrent dataset counts:")
    for class_name in CLASSES:
        print(f"  {class_name}: {counts.get(class_name, 0)}")


def main():
    parser = argparse.ArgumentParser(description="Capture fixed-ROI images for the dataset.")
    parser.add_argument("--class-name", required=True, choices=CLASSES, help="Class folder to save into.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index. Try 0, 1, or 2.")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.5,
        help="Scale factor for the preview window. Example: 1.5 or 2.0.",
    )
    parser.add_argument("--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Optional number of NEW images to capture in this session. 0 means manual unlimited.",
    )
    args = parser.parse_args()

    roi = parse_roi(args.roi)

    save_dir = DATASET_DIR / args.class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}. Try --camera-index 1 or 2.")
        return

    session_saved = 0
    counts = get_class_counts()
    next_filename, next_number = get_next_filename(save_dir, args.class_name)

    window_name = "Capture ROI Dataset Images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(args.width * args.display_scale), int(args.height * args.display_scale))

    print(f"Saving cropped ROI images to: {save_dir}")
    print(f"Class label: {args.class_name}")
    print(f"Using ROI: {roi}")
    print(f"Display scale: {args.display_scale}")
    print("Controls: c or SPACE = save ROI crop, q = quit")
    print_counts(counts)
    print(f"Next image will be: {next_filename.name}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame.")
            break

        display = frame.copy()
        draw_roi(display, roi)
        draw_counts_panel(display, counts, args.class_name, next_number)

        if args.display_scale != 1.0:
            display = cv2.resize(display, None, fx=args.display_scale, fy=args.display_scale)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("c") or key == 32:
            roi_crop = crop_roi(frame, roi)
            next_filename, next_number = get_next_filename(save_dir, args.class_name)
            cv2.imwrite(str(next_filename), roi_crop)

            session_saved += 1
            counts = get_class_counts()

            print(f"Saved ROI crop: {next_filename}")
            print_counts(counts)

            next_filename, next_number = get_next_filename(save_dir, args.class_name)
            print(f"Next image will be: {next_filename.name}")

            if args.count > 0 and session_saved >= args.count:
                print(f"Reached requested session count: {args.count}")
                break

    cap.release()
    cv2.destroyAllWindows()

    final_counts = get_class_counts()
    print("\nCapture session finished.")
    print(f"New images captured in this session for '{args.class_name}': {session_saved}")
    print_counts(final_counts)


if __name__ == "__main__":
    main()
