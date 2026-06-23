# Source Files

This folder contains all Python scripts for the lane detection pipeline.

---

## `lane_detection_image.py`

Runs lane detection on a **single static image**.

Loads one road image, processes it through the full classical CV pipeline, draws averaged green lane lines on top, displays the result in a window, and saves the annotated image to the `outputs/` folder.

**Use this for:** testing and tuning the pipeline on a still image before applying it to video.

**To run:**
```bash
python src/lane_detection_image.py
```

**Input:** `images/` folder  
**Output:** `outputs/lane_detect_output.jpg`

---

## `lane_detection_video.py`

Runs lane detection on a **video file in real time**.

Reads the video frame by frame, applies the full pipeline to each frame, draws two solid averaged green lines (one per lane), and writes the annotated video to the `outputs/` folder. Uses temporal smoothing — averages lane positions over the last 10 frames — to keep the lines stable and prevent flickering.

Press **Q** to quit the video window early.

**Use this for:** real-time lane detection on dashcam footage.

**To run:**
```bash
python src/lane_detection_video.py
```

**Input:** `videos/` folder  
**Output:** `outputs/dashcam4_output.mp4`

---

## `find_coords.py`

A **coordinate helper tool** for tuning the Region of Interest (ROI) mask.

The ROI mask is a trapezoid drawn over the road area. Every time you use a new image or video with a different camera angle or resolution, the trapezoid needs to be retuned to match the new road position on screen.

This tool opens any image and lets you click directly on it. Every click prints the exact pixel coordinates `(x, y)` to the terminal. You then copy those coordinates into the trapezoid definition in `lane_detect.py` or `lane_video.py`.

**How to use:**
1. Update the image path inside `find_coords.py` to point to your image or video frame screenshot
2. Run the script
3. Click on these 4 spots on the road:
   - Bottom-left: where the left lane line meets the bottom of the frame
   - Bottom-right: where the right lane line meets the bottom of the frame
   - Top-right: where the right lane line vanishes near the horizon
   - Top-left: where the left lane line vanishes near the horizon
4. The terminal prints each clicked coordinate
5. Copy the 4 coordinates into the trapezoid array in your detection script

**To run:**
```bash
python src/find_coords.py
```

**Press any key** to close the window when done.

> **Note:** If the image is larger than your screen, resize it inside `find_coords.py` by adding:
> ```python
> image = cv2.resize(image, (1280, 720))
> ```
> This ensures the coordinates match the resized frame used during detection.
