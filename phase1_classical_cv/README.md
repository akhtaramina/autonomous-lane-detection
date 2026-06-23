# Classical CV Lane Detection

A real-time lane detection system built using classical computer vision techniques (OpenCV). This is Phase 1 of an autonomous lane-following system being developed for deployment on a Zybo Z7-10 FPGA board.

## Skills Demonstrated

- OpenCV
- NumPy
- Image Processing
- Edge Detection
- Hough Transform
- Geometric Reasoning
- Real-Time Video Processing
- Algorithm Design
- Parameter Tuning

## Demo

![Lane Detection Output](outputs/lane_detect_output.jpg)

## Pipeline

Every frame goes through this pipeline:

```
Raw Frame → Grayscale → Gaussian Blur → Canny Edge Detection → ROI Mask → Hough Transform → Line Averaging → Output
```

## Project Structure

```
├── src/
│   ├── lane_detection_image.py      # Static image lane detection
│   ├── lane_detection_video.py      # Real-time video lane detection
│   └── find_coords.py               # Helper tool for ROI coordinate tuning
├── images/                          # Test images
├── videos/                          # Demo input assets and test videos
├── outputs/                         # Annotated output images and videos
└── README.md
```

## How It Works

**Grayscale Conversion** — strips color information since lane detection only needs brightness contrast.

**Gaussian Blur** — smooths pixel-level noise before edge detection to reduce false edges.

**Canny Edge Detection** — finds sharp brightness changes (lane markings against dark road) using a two-threshold system.

**Region of Interest Mask** — crops the image to a trapezoid covering only the road area, eliminating sky and roadside noise.

**Hough Line Transform** — votes on which straight lines best explain the detected edge points.

**Line Averaging** — separates left/right lanes by slope sign and x-position, fits a single clean line per side using `np.polyfit`.

**Temporal Smoothing** — averages lane positions over the last 10 frames using a deque to stabilize output across video frames.

## Setup

```bash
pip install opencv-python numpy
```

## Usage

**Static image:**
```bash
python src/lane_detection_image.py
```

**Video:**
```bash
python src/lane_detection_video.py
```

## Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| Gaussian kernel | (5, 5) | Noise suppression |
| Canny low threshold | 50 | Weak edge detection |
| Canny high threshold | 150 | Strong edge detection |
| Hough votes | 30 | Minimum line votes |
| Min line length | 20px | Filter short segments |
| Max line gap | 100px | Connect broken lines |
| Slope filter | 0.5 – 1.2 | Remove noise lines |
| Temporal window | 10 frames | Smoothing history |

## Limitations

- Hough Transform detects straight lines only — curves reduce accuracy
- Parameters are tuned per video — not generalizable across different road conditions
- Low contrast lane markings reduce detection reliability

These limitations motivate the transition to deep learning in Phase 2.

## Results

Successfully detects and tracks lane boundaries in highway driving footage using only classical computer vision techniques.

Features:

- Real-time lane detection
- Left/right lane separation
- Lane extrapolation
- Temporal smoothing across frames
- Video output generation

## Environment

- Python 3.14.2
- OpenCV 4.13.0
- NumPy
- Windows 11
