import cv2
import numpy as np
from pathlib import Path

LOOKAHEAD_RATIO = 0.85
MIN_SEGMENT_WIDTH = 20

LOWER_WHITE = np.array([0, 0, 180], dtype=np.uint8)
UPPER_WHITE = np.array([180, 80, 255], dtype=np.uint8)

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "assets" / "straight_continuous.jpg"
OUTPUT_PATH = BASE_DIR / "outputs" / "straight_continuous_lane_center.jpg"


def create_white_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)
    return cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
    # return cv2.inRange(hsv, lower_white, upper_white)


def find_white_segments_on_row(mask, row_y):
    row = mask[row_y, :]

    white_x_positions = np.where(row == 255)[0]

    if len(white_x_positions) == 0:
        return []

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

    return segments


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Could not load image:\n{IMAGE_PATH}")

    h, w = image.shape[:2]

    mask = create_white_mask(image)

    # lookahead_y = int(h * 0.85)
    lookahead_y = int(h * LOOKAHEAD_RATIO)

    segments = find_white_segments_on_row(mask, lookahead_y)

    # Keep only reasonably wide white segments to ignore tiny noise.
    # MIN_SEGMENT_WIDTH = 20
    segments = [
        (start, end)
        for start, end in segments
        if (end - start) >= MIN_SEGMENT_WIDTH
    ]

    print(f"Image width       : {w}")
    print(f"Image height      : {h}")
    print(f"Lookahead row y   : {lookahead_y}")
    print(f"Segments detected : {segments}")

    if len(segments) < 2:
        print("Could not detect both tape boundaries.")
        return

    left_segment = segments[0]
    right_segment = segments[-1]

    left_x = (left_segment[0] + left_segment[1]) // 2
    right_x = (right_segment[0] + right_segment[1]) // 2

    lane_center_x = (left_x + right_x) // 2
    vehicle_center_x = w // 2
    offset = lane_center_x - vehicle_center_x

    print(f"Left tape x       : {left_x}")
    print(f"Right tape x      : {right_x}")
    print(f"Lane center x     : {lane_center_x}")
    print(f"Vehicle center x  : {vehicle_center_x}")
    print(f"Offset            : {offset}")

    output = image.copy()

    # Draw lookahead row.
    cv2.line(output, (0, lookahead_y), (w, lookahead_y), (0, 255, 0), 4)

    # Draw tape centers at lookahead row.
    cv2.circle(output, (left_x, lookahead_y), 12, (0, 0, 255), -1)
    cv2.circle(output, (right_x, lookahead_y), 12, (0, 0, 255), -1)

    # Draw lane center and vehicle center.
    cv2.circle(output, (lane_center_x, lookahead_y), 14, (0, 255, 255), -1)
    cv2.line(output, (lane_center_x, h), (lane_center_x, h - 250), (0, 255, 255), 6)
    cv2.line(output, (vehicle_center_x, h), (vehicle_center_x, h - 250), (255, 0, 0), 6)

    cv2.imwrite(str(OUTPUT_PATH), output)

    print(f"Saved output image to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()