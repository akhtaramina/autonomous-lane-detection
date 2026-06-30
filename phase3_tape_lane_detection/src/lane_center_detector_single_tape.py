import cv2
import numpy as np
from pathlib import Path
from collections import deque

class LaneWidthEstimator:
    
    def __init__(self, buffer_size=30):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, left_x, right_x):
        width = right_x - left_x
        self.buffer.append(width)

    def estimate(self):
        if len(self.buffer) == 0:
            return None
        return int(np.mean(self.buffer))
    
# ==========================================================
# Detection Parameters
# ==========================================================

ROI_START_RATIO = 0.55
LOOKAHEAD_RATIO = 0.45
SCAN_BAND_HEIGHT = 120
MIN_SEGMENT_WIDTH = 20

LOWER_WHITE = np.array([0, 0, 120], dtype=np.uint8)
UPPER_WHITE = np.array([180, 120, 255], dtype=np.uint8)

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
    return cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)


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
            sstart_x = white_x_positions[0]
            previous_x = x

    segments.append((start_x, previous_x))

    return segments, band_y1, band_y2


def process_frame(frame, estimator):
    h, w = frame.shape[:2]

    roi_start = int(h * ROI_START_RATIO)
    roi = frame[roi_start:, :]

    roi_h, roi_w = roi.shape[:2]

    mask = create_white_mask(roi)

    lookahead_y_roi = int(roi_h * LOOKAHEAD_RATIO)

    segments, band_y1_roi, band_y2_roi = find_white_segments_in_band(
        mask,
        lookahead_y_roi,
        SCAN_BAND_HEIGHT
    )

    segments = [
        (start, end)
        for start, end in segments
        if (end - start) >= MIN_SEGMENT_WIDTH
    ]

    output = frame.copy()

    # Convert ROI y-coordinates back to full-frame y-coordinates
    lookahead_y_frame = lookahead_y_roi + roi_start
    band_y1_frame = band_y1_roi + roi_start
    band_y2_frame = band_y2_roi + roi_start

    # Draw ROI boundary
    cv2.rectangle(
        output,
        (0, roi_start),
        (w, h),
        (0, 180, 0),
        3
    )

    # Draw scan band
    cv2.rectangle(
        output,
        (0, band_y1_frame),
        (w, band_y2_frame),
        (0, 255, 0),
        4
    )

    # Sort by width, take two largest, then sort left to right
    segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)
    segments = sorted(segments[:2], key=lambda s: s[0])

    if len(segments) == 0:
        # No tape detected at all — nothing we can do
        print("No tape detected.")
        return output, None

    elif len(segments) == 1:
        # Only one tape visible — try to recover using estimated lane width
        estimated_width = estimator.estimate() if estimator is not None else None

        if estimated_width is None:
            print("Single tape detected but no width estimate available yet.")
            return output, None

        detected_x = (segments[0][0] + segments[0][1]) // 2
        frame_center = w // 2

        if detected_x < frame_center:
            # Left tape detected, estimate right
            left_x = detected_x
            right_x = left_x + estimated_width
            print(f"Left tape only. Estimating right at x={right_x}")
        else:
            # Right tape detected, estimate left
            right_x = detected_x
            left_x = right_x - estimated_width
            print(f"Right tape only. Estimating left at x={left_x}")

    else:
        # Both tapes detected normally
        left_segment = segments[0]
        right_segment = segments[1]

        left_x = (left_segment[0] + left_segment[1]) // 2
        right_x = (right_segment[0] + right_segment[1]) // 2

        if estimator is not None:
            estimator.update(left_x, right_x)

    lane_center_x = (left_x + right_x) // 2
    vehicle_center_x = w // 2
    offset = lane_center_x - vehicle_center_x

    # Draw tape centers — orange if estimated, red if detected
    if len(segments) == 1:
        if detected_x < w // 2:
            # Left is real, right is estimated
            cv2.circle(output, (left_x, lookahead_y_frame), 12, (0, 0, 255), -1)
            cv2.circle(output, (right_x, lookahead_y_frame), 12, (0, 165, 255), -1)
        else:
            # Right is real, left is estimated
            cv2.circle(output, (left_x, lookahead_y_frame), 12, (0, 165, 255), -1)
            cv2.circle(output, (right_x, lookahead_y_frame), 12, (0, 0, 255), -1)
    else:
        cv2.circle(output, (left_x, lookahead_y_frame), 12, (0, 0, 255), -1)
        cv2.circle(output, (right_x, lookahead_y_frame), 12, (0, 0, 255), -1)

    # Draw lane center
    cv2.circle(output, (lane_center_x, lookahead_y_frame), 14, (0, 255, 255), -1)
    cv2.line(output, (lane_center_x, h), (lane_center_x, h - 250), (0, 255, 255), 6)

    # Draw vehicle center
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

    output, offset = process_frame(image, estimator=None)

    cv2.imwrite(str(OUTPUT_PATH), output)

    print(f"Saved output image to:\n{OUTPUT_PATH}")
    print(f"Final offset: {offset}")


if __name__ == "__main__":
    main()