import cv2
import time
import numpy as np
import onnxruntime as ort
import scipy.special

BASE_DIR = r"C:\Users\amina\autonomous_lane_detect"

VIDEO_PATH = BASE_DIR + r"\videos\dashcam4.mp4"
MODEL_PATH = BASE_DIR + r"\models\tusimple_18.onnx"
OUTPUT_PATH = BASE_DIR + r"\outputs\dashcam4_onnx_output_100frames.mp4"

MODEL_INPUT_WIDTH = 800
MODEL_INPUT_HEIGHT = 288

OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

GRIDDING_NUM = 100
CLS_NUM_PER_LANE = 56
MAX_FRAMES = 100

ROW_ANCHOR = [
    64, 68, 72, 76, 80, 84, 88, 92,
    96, 100, 104, 108, 112, 116, 120, 124,
    128, 132, 136, 140, 144, 148, 152, 156,
    160, 164, 168, 172, 176, 180, 184, 188,
    192, 196, 200, 204, 208, 212, 216, 220,
    224, 228, 232, 236, 240, 244, 248, 252,
    256, 260, 264, 268, 272, 276, 280, 284
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame):
    image = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)


def draw_lanes(frame, output):
    h, w = frame.shape[:2]

    output = output[0]
    output = output[:, ::-1, :]

    prob = scipy.special.softmax(output[:-1, :, :], axis=0)

    idx = np.arange(GRIDDING_NUM) + 1
    idx = idx.reshape(-1, 1, 1)

    loc = np.sum(prob * idx, axis=0)
    output_argmax = np.argmax(output, axis=0)

    loc[output_argmax == GRIDDING_NUM] = 0

    col_sample = np.linspace(0, MODEL_INPUT_WIDTH - 1, GRIDDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    for lane_index in range(loc.shape[1]):
        points = []

        if np.sum(loc[:, lane_index] != 0) > 2:
            for point_index in range(loc.shape[0]):
                if loc[point_index, lane_index] > 0:
                    x = int(
                        loc[point_index, lane_index]
                        * col_sample_w
                        * w
                        / MODEL_INPUT_WIDTH
                    ) - 1

                    y = int(
                        h
                        * ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - point_index]
                        / MODEL_INPUT_HEIGHT
                    ) - 1

                    points.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

    return frame


def main():
    print("Loading ONNX model...")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video FPS: {fps}")
    print(f"Original video size: {original_width} x {original_height}")
    print(f"Total frames: {total_frames}")
    print(f"Processing only first {MAX_FRAMES} frames")
    print(f"Output video size: {OUTPUT_WIDTH} x {OUTPUT_HEIGHT}")

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    )

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count >= MAX_FRAMES:
            break

        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        input_tensor = preprocess(frame)
        output = session.run([output_name], {input_name: input_tensor})[0]

        result = draw_lanes(frame, output)
        writer.write(result)

        frame_count += 1

        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            processing_fps = frame_count / elapsed
            print(
                f"Processed {frame_count}/{MAX_FRAMES} frames "
                f"| Speed: {processing_fps:.2f} FPS"
            )

    cap.release()
    writer.release()

    print("Done.")
    print(f"Saved output video to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()