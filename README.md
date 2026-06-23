# Autonomous Lane Detection System

This project is a long-term autonomous lane-following system built as a Computer Engineering project.

## Current Status
- Phase 1: Classical CV lane detection (completed)
- Phase 2: ONNX lane detection (current)
- Static image inference working
- 100-frame video inference working

## Model Setup

This project uses the Ultra Fast Lane Detection ONNX model.
Place the downloaded model file at:
models/tusimple_18.onnx
See models/README.md for details and download instructions.


## Current Scripts
### run_image_onnx.py

Runs Ultra Fast Lane Detection on a single image.

Input:

```text id="j9m0bo"
images/highway.jpg
```

Output:

```text id="mjlwm8"
outputs/highway.jpg
```

Purpose:

* Verify model loading
* Verify preprocessing
* Verify lane decoding
* Debug lane detection on a single frame

---

### run_video_onnx.py

Runs Ultra Fast Lane Detection on a video.

Input:

```text id="ol9p8k"
videos/dashcam.mp4
```

Output:

```text id="eqk6hy"
outputs/dashcam_onnx_output.mp4
```

Current implementation:

* CPU inference using ONNX Runtime
* Processes first 100 frames for faster debugging
* Draws detected lane overlays on video frames

Purpose:

* Validate lane detection across multiple frames
* Evaluate lane stability
* Prepare for future lane-center estimation and steering control
