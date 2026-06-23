from pathlib import Path
import cv2

# Get the project root (one level up from src/)
BASE_DIR   = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "images" / "test_road1.jpg"

image = cv2.imread(str(IMAGE_PATH))
image = cv2.resize(image, (1280, 720))

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: ({x}, {y})")

cv2.imshow("Click to find coordinates", image)
cv2.setMouseCallback("Click to find coordinates", click)
cv2.waitKey(0)