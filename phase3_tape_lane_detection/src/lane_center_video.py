import cv2
from pathlib import Path

from lane_center_detector import process_frame


BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "assets" / "tape_lane_video.mp4"
OUTPUT_PATH = BASE_DIR / "outputs" / "tape_lane_video_output.mp4"

MAX_FRAMES = 300


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video:\n{VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video path   : {VIDEO_PATH}")
    print(f"FPS          : {fps}")
    print(f"Frame size   : {width} x {height}")
    print(f"Total frames : {total_frames}")
    print(f"Max frames   : {MAX_FRAMES}")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count >= MAX_FRAMES:
            break

        output_frame, offset = process_frame(frame)

        writer.write(output_frame)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | Offset: {offset}")

    cap.release()
    writer.release()

    print("Done.")
    print(f"Saved output video to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()