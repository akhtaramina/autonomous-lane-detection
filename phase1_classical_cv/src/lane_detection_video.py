from pathlib import Path
import cv2
import numpy as np
from collections import deque

# Get the project root (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH  = BASE_DIR / "videos" / "dashcam.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "dashcam_output.mp4"


def average_lines(frame, lines):
    left_points = []
    right_points = []
    height, width = frame.shape[:2]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        midx = (x1 + x2) / 2
        if abs(slope) < 0.5 or abs(slope) > 1.2:
            continue
        if slope < 0 and midx < width * 0.55:
            left_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and midx > width * 0.45:
            right_points.extend([(x1, y1), (x2, y2)])

    result = []
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


# Temporal smoothing history
left_history  = deque(maxlen=10)
right_history = deque(maxlen=10)

cap = cv2.VideoCapture(str(VIDEO_PATH))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, 30, (1280, 720))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    height, width = frame.shape[:2]

    gray         = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred      = cv2.GaussianBlur(gray, (5, 5), 0)
    edges        = cv2.Canny(blurred, 50, 150)

    mask = np.zeros_like(edges)
    trapezoid = np.array([[
        (172, 717),
        (1273, 700),
        (900, 503),
        (380, 501)
    ]])
    cv2.fillPoly(mask, trapezoid, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30,
                            minLineLength=20, maxLineGap=100)

    if lines is not None:
        averaged = average_lines(frame, lines)
        if len(averaged) >= 1:
            left_history.append(averaged[0])
        if len(averaged) >= 2:
            right_history.append(averaged[1])

    if left_history:
        x1 = int(np.mean([c[0] for c in left_history]))
        y1 = int(np.mean([c[1] for c in left_history]))
        x2 = int(np.mean([c[2] for c in left_history]))
        y2 = int(np.mean([c[3] for c in left_history]))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if right_history:
        x1 = int(np.mean([c[0] for c in right_history]))
        y1 = int(np.mean([c[1] for c in right_history]))
        x2 = int(np.mean([c[2] for c in right_history]))
        y2 = int(np.mean([c[3] for c in right_history]))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("Lane Detection Video", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()