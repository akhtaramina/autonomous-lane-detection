import cv2
import numpy as np
from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = BASE_DIR / "assets" / "straight_continuous.jpg"
OUTPUT_PATH = BASE_DIR / "outputs" / "straight_continuous_mask.jpg"

#bottom x-position of each tape boundary  
# def get_bottom_center_x(contour):
#     x, y, w, h = cv2.boundingRect(contour)
#     center_x = x + w // 2
#     bottom_y = y + h
#     return center_x, bottom_y

def get_bottom_center_x(contour):
    points = contour.reshape(-1, 2)

    max_y = np.max(points[:, 1])

    bottom_points = points[points[:, 1] == max_y]

    center_x = int(np.mean(bottom_points[:, 0]))
    bottom_y = int(max_y)

    return center_x, bottom_y


# ==========================================================
# Main
# ==========================================================

def main():
    # Load image
    image = cv2.imread(str(IMAGE_PATH))

    # check the image dimension
    h, w = image.shape[:2]
    print(f"Image width : {w}")
    print(f"Image height: {h}")

    if image is None:
        raise FileNotFoundError(f"Could not load image:\n{IMAGE_PATH}")

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White threshold
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 80, 255], dtype=np.uint8)

    # Binary mask
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Contours found: {len(contours)}")

    contour_image = image.copy()
    
    # Draw contours
    MIN_CONTOUR_AREA = 1000

    large_contours = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > MIN_CONTOUR_AREA:
            large_contours.append(contour)

    print(f"Large contours: {len(large_contours)}")
    
    # Sort large contours from left to right using their bounding boxes
    large_contours = sorted(
        large_contours,
        key=lambda contour: cv2.boundingRect(contour)[0] #OpenCV draws the smallest rectangle that completely contains the large contoure
    )

    if len(large_contours) >= 2:
        left_tape = large_contours[0]
        right_tape = large_contours[-1]


        left_x, left_y = get_bottom_center_x(left_tape)
        right_x, right_y = get_bottom_center_x(right_tape)

        print(f"Left tape bottom center : ({left_x}, {left_y})")
        print(f"Right tape bottom center: ({right_x}, {right_y})")
        
        #calculate lane center
        lane_center_x = (left_x + right_x) // 2

        h, w = image.shape[:2]
        vehicle_center_x = w // 2

        offset = lane_center_x - vehicle_center_x

        print(f"Lane center x   : {lane_center_x}")
        print(f"Vehicle center x: {vehicle_center_x}")
        print(f"Offset          : {offset}")
        
        #draw lines
        cv2.circle(contour_image, (lane_center_x, left_y), 10, (0, 255, 255), -1)
        cv2.line(contour_image, (lane_center_x, h), (lane_center_x, h - 150), (0, 255, 255), 4)
        cv2.line(contour_image, (vehicle_center_x, h), (vehicle_center_x, h - 150), (255, 0, 0), 4)

        # print("Left and right tape detected.")
    else:
        print("Could not detect both tape boundaries.")

    cv2.drawContours(
        contour_image,
        large_contours,
        -1,
        (0, 0, 255),
        3
    )

    cv2.imwrite(str(OUTPUT_PATH), contour_image)

    print(f"Saved contour image to:\n{OUTPUT_PATH}")

# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()