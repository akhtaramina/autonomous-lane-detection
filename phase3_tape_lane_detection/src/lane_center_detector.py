import cv2
import numpy as np
from pathlib import Path


# ==========================================================
# Detection Parameters
# ==========================================================

LOOKAHEAD_RATIO = 0.85
SCAN_BAND_HEIGHT = 80
MIN_SEGMENT_WIDTH = 20

LOWER_WHITE = np.array([0, 0, 180], dtype=np.uint8)
UPPER_WHITE = np.array([180, 80, 255], dtype=np.uint8)


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "assets" / "straight_continuous.jpg"
OUTPUT_PATH = BASE_DIR / "outputs" / "straight_continuous_lane_center.jpg"


# ==========================================================
# Helper Functions
# ==========================================================

def create_white_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    return mask


def find_white_segments_in_band(mask, center_y, band_height):
    h, w = mask.shape[:2]

    half_band = band_height // 2

    band_y1 = max(0, center_y - half_band)
    band_y2 = min(h, center_y + half_band)

    band = mask[band_y1:band_y2, :]

    column_scores = np.sum(band == 255, axis=0)

    min_white_pixels = max(1, band_height // 4)

    white_x_positions = np.where(column_scores >= min_white_pixels)[0]

    if len(white_x_positions) == 0:
        return [], band_y1, band_y2

    segments = []
    start_x = white_x_positions[0]
    previous_x = white_x_positions[0]

    for x in white_x_positions[1:]:
        if x == previous_x + 1:
            previous_x = x
        else:
            segments.append((start_x, previous_x))
            start_x = x
            previous_x = x

    segments.append((start_x, previous_x))

    return segments, band_y1, band_y2

def process_frame(frame):
    h, w = frame.shape[:2]

    mask = create_white_mask(frame)

    lookahead_y = int(h * LOOKAHEAD_RATIO)

    segments, band_y1, band_y2 = find_white_segments_in_band(
        mask,
        lookahead_y,
        SCAN_BAND_HEIGHT
    )

    segments = [
        (start, end)
        for start, end in segments
        if (end - start) >= MIN_SEGMENT_WIDTH
    ]

    output = frame.copy()

    cv2.rectangle(
        output,
        (0, band_y1),
        (w, band_y2),
        (0, 255, 0),
        4
    )

    if len(segments) < 2:
        print("Could not detect both tape boundaries.")
        return output, None

    left_segment = segments[0]
    right_segment = segments[-1]

    left_x = (left_segment[0] + left_segment[1]) // 2
    right_x = (right_segment[0] + right_segment[1]) // 2

    lane_center_x = (left_x + right_x) // 2
    vehicle_center_x = w // 2
    offset = lane_center_x - vehicle_center_x

    cv2.circle(output, (left_x, lookahead_y), 12, (0, 0, 255), -1)
    cv2.circle(output, (right_x, lookahead_y), 12, (0, 0, 255), -1)

    cv2.circle(output, (lane_center_x, lookahead_y), 14, (0, 255, 255), -1)

    cv2.line(output, (lane_center_x, h), (lane_center_x, h - 250), (0, 255, 255), 6)
    cv2.line(output, (vehicle_center_x, h), (vehicle_center_x, h - 250), (255, 0, 0), 6)

    print(f"Segments detected : {segments}")
    print(f"Left tape x       : {left_x}")
    print(f"Right tape x      : {right_x}")
    print(f"Lane center x     : {lane_center_x}")
    print(f"Vehicle center x  : {vehicle_center_x}")
    print(f"Offset            : {offset}")

    return output, offset
# ==========================================================
# Main
# ==========================================================

def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Could not load image:\n{IMAGE_PATH}")

    print(f"Input image: {IMAGE_PATH.name}")
    print(f"Image size : {image.shape[1]} x {image.shape[0]}")

    output, offset = process_frame(image)

    cv2.imwrite(str(OUTPUT_PATH), output)

    print(f"Saved output image to:\n{OUTPUT_PATH}")
    print(f"Final offset: {offset}")

if __name__ == "__main__":
    main()