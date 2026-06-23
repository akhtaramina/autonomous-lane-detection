from pathlib import Path
import cv2
import numpy as np

# Get the project root (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH   = BASE_DIR / "images" / "test_road1.jpg"
OUTPUT_IMAGE = BASE_DIR / "outputs" / "lane_detect_output.jpg"


def average_lines(frame, lines):
    left_points  = []
    right_points = []
    height, width = frame.shape[:2]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        midx  = (x1 + x2) / 2

        # Filter out horizontal and nearly vertical lines
        if abs(slope) < 0.5 or abs(slope) > 1.2:
            continue

        if slope < 0 and midx < width * 0.5:
            left_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and midx > width * 0.5:
            right_points.extend([(x1, y1), (x2, y2)])

    result = []
    height = frame.shape[0]

    for points in [left_points, right_points]:
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        m, b = np.polyfit(xs, ys, 1)
        y1 = height
        y2 = int(height * 0.75)
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        result.append((x1, y1, x2, y2))

    result.sort(key=lambda coords: coords[0])
    return result


# Load image
image = cv2.imread(str(IMAGE_PATH))
image = cv2.resize(image, (1280, 720))
height, width = image.shape[:2]

# Step 1 - Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2 - Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3 - Edges
edges = cv2.Canny(blurred, 50, 150)

# Step 4 - Mask
mask = np.zeros_like(edges)
trapezoid = np.array([[
    (172, 717),
    (1273, 700),
    (900, 503),
    (380, 501)
]])
cv2.fillPoly(mask, trapezoid, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# Step 5 - Hough lines
lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30,
                        minLineLength=20, maxLineGap=50)

# Step 6 - Draw averaged lines
line_image = image.copy()

if lines is not None:
    averaged = average_lines(image, lines)
    for x1, y1, x2, y2 in averaged:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

cv2.imshow("Lane Detection", line_image)
cv2.imwrite(str(OUTPUT_IMAGE), line_image)
cv2.waitKey(0)