import cv2
import numpy as np
import onnxruntime as ort
import scipy.special

BASE_DIR = r"C:\Users\amina\autonomous_lane_detect"

IMAGE_PATH = BASE_DIR + r"\images\highway.jpg"
MODEL_PATH = BASE_DIR + r"\models\tusimple_18.onnx"
OUTPUT_PATH = BASE_DIR + r"\outputs\highway.jpg"

INPUT_WIDTH = 800
INPUT_HEIGHT = 288
GRIDDING_NUM = 100
CLS_NUM_PER_LANE = 56

ROW_ANCHOR = [
    64, 68, 72, 76, 80, 84, 88, 92,
    96, 100, 104, 108, 112, 116, 120, 124,
    128, 132, 136, 140, 144, 148, 152, 156,
    160, 164, 168, 172, 176, 180, 184, 188,
    192, 196, 200, 204, 208, 212, 216, 220,
    224, 228, 232, 236, 240, 244, 248, 252,
    256, 260, 264, 268, 272, 276, 280, 284
]

MEAN = np.array([0.406, 0.456, 0.485], dtype=np.float32)
STD = np.array([0.225, 0.224, 0.229], dtype=np.float32)


def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def draw_lanes(original_image, output):
    h, w = original_image.shape[:2]

    output = output[0]
    output = output[:, ::-1, :]

    prob = scipy.special.softmax(output[:-1, :, :], axis=0)

    idx = np.arange(GRIDDING_NUM) + 1
    idx = idx.reshape(-1, 1, 1)

    loc = np.sum(prob * idx, axis=0)
    output_argmax = np.argmax(output, axis=0)

    loc[output_argmax == GRIDDING_NUM] = 0

    col_sample = np.linspace(0, INPUT_WIDTH - 1, GRIDDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    for lane_index in range(loc.shape[1]):
        points = []

        if np.sum(loc[:, lane_index] != 0) > 2:
            for point_index in range(loc.shape[0]):
                if loc[point_index, lane_index] > 0:
                    x = int(loc[point_index, lane_index] * col_sample_w * w / INPUT_WIDTH) - 1
                    y = int(h * ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - point_index] / INPUT_HEIGHT) - 1

                    points.append((x, y))
                    cv2.circle(original_image, (x, y), 5, (0, 255, 0), -1)

        for i in range(len(points) - 1):
            cv2.line(original_image, points[i], points[i + 1], (0, 255, 0), 3)

    return original_image


def main():
    print("Loading image...")
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    print("Loading ONNX model...")
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Running inference...")
    input_tensor = preprocess(image)
    output = session.run([output_name], {input_name: input_tensor})[0]

    print("Drawing lanes...")
    result = draw_lanes(image, output)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Saved output image to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()