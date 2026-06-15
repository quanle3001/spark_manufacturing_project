import cv2

from config import ROI, ROI_COLOR_BLUE


def parse_roi(values):
    """Convert a list/tuple of four values into an ROI tuple."""
    if values is None:
        return ROI

    if len(values) != 4:
        raise ValueError("ROI must contain exactly four numbers: x1 y1 x2 y2")

    x1, y1, x2, y2 = [int(v) for v in values]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid ROI. Make sure x2 > x1 and y2 > y1.")

    return x1, y1, x2, y2


def clamp_roi_to_frame(frame, roi):
    """Make sure ROI coordinates stay inside the camera frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi

    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if x2 <= x1:
        x1, x2 = 0, w
    if y2 <= y1:
        y1, y2 = 0, h

    return x1, y1, x2, y2


def crop_roi(frame, roi):
    """Return only the image inside the fixed ROI."""
    x1, y1, x2, y2 = clamp_roi_to_frame(frame, roi)
    return frame[y1:y2, x1:x2]


def draw_roi(frame, roi, color=ROI_COLOR_BLUE, thickness=2):
    """Draw the fixed ROI rectangle on a frame."""
    x1, y1, x2, y2 = clamp_roi_to_frame(frame, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def maybe_crop_full_frame_image(image, roi):
    """
    If an image looks like a full camera frame, crop the ROI.
    If the image is already an ROI crop, leave it alone.

    This makes training safer if you accidentally save full frames.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi

    if w >= x2 and h >= y2:
        return crop_roi(image, roi)

    return image
