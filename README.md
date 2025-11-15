# Traffic Density Estimation — YOLOv8

Project: Real-time traffic density estimation using a fine-tuned YOLOv8 model.

## Project Overview
This repository implements a real-time Traffic Density Estimation pipeline using a fine-tuned YOLOv8 model. It detects vehicles in video frames, counts vehicles in designated lane regions, and annotates traffic intensity (e.g., Smooth / Heavy). The system is intended for experimentation, prototyping, and deployment in local setups or edge devices.

## Dataset Description
- Name: Top-View Vehicle Detection Image Dataset for YOLOv8
- Classes: single combined class "Vehicle" (cars, trucks, buses)
- Total images: 626 (640×640 px)
- Split: 536 train / 90 validation
- Annotations: YOLO format (xmin_center_y_center_width_height normalized)
- Source & access:
  - Roboflow: https://universe.roboflow.com/farzad/vehicle_detection_yolov8
  - Kaggle: https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset
- Preprocessing: uniform resizing (640×640), optional augmentations (flips, etc.)

## Features
- Real-time vehicle detection with YOLOv8.
- Per-lane vehicle counting using configurable ROIs and thresholds.
- Traffic intensity labeling (Smooth / Heavy) per lane.
- Visual overlays (lane polygons, counts, intensity banners).
- Export-friendly model formats (PyTorch `.pt`, optional `.onnx`).

## Technologies Used
- Python 3.8+
- OpenCV (cv2) — video I/O, drawing, visualization
- NumPy — numeric operations and polygon definitions
- Ultralytics YOLO (YOLOv8) — detection model and inference API
- (Optional) PyTorch with CUDA for GPU acceleration

## Project Structure
- c:\Traffic_Density_Estimation\
  - real_time_traffic_analysis.py — main inference script (real-time processing)
  - models/best.pt — fine-tuned YOLOv8 model (not included)
  - sample_video.mp4 — sample input video (user-supplied)
  - processed_sample_video.avi — output written by the script (runtime)
  - real-time_traffic_density_estimation_yolov8.ipynb — development notebook
  - images/ — cover and sample images
  - Running_Real-Time_Traffic_Analysis.gif — demo GIF
  - README.md — this file

## Installation & Setup
1. Clone the repository:
   git clone https://github.com/FarzadNekouee/YOLOv8_Traffic_Density_Estimation.git
2. Create and activate a Python environment:
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
   or at minimum:
   pip install ultralytics opencv-python numpy
4. (Optional, recommended) Install GPU-backed PyTorch compatible with your CUDA version as per https://pytorch.org.

## Usage Guide
1. Place your trained model at `models/best.pt` (update path in script if different).
2. Put the input video at `sample_video.mp4` or edit `real_time_traffic_analysis.py` to point to your file.
3. Run the analysis:
   python real_time_traffic_analysis.py
4. Controls:
   - Press `q` on the display window to quit early.
5. Output:
   - The script writes `processed_sample_video.avi` and displays annotated frames in a window.

## Configuration (script variables and recommended tuning)
Edit `real_time_traffic_analysis.py` to change defaults:

- Model and inference:
  - model path: YOLO('models/best.pt')
  - imgsz (in predict): default 640 — reduce to 320 for higher FPS with reduced accuracy
  - conf (in predict): default 0.4 — lower to detect smaller objects, raise to reduce false positives
- Video I/O:
  - input: cv2.VideoCapture('sample_video.mp4')
  - output: filename and codec in VideoWriter
- ROI and lanes:
  - vertical slice: x1, x2 = 325, 635 (rows used to zero-out outside region)
  - lane_threshold = 609 (x-coordinate used to split left/right lanes)
  - vertices1, vertices2 — polygon vertex arrays defining lane overlays
- Counting logic:
  - heavy_traffic_threshold = 10 — number of vehicles above which lane labeled "Heavy"

Suggested improvements to configuration:
- Use polygon point-in-polygon instead of a simple x-threshold for robust lane assignment.
- Add class filters (cars, buses, trucks) if model differentiates classes.
- Integrate a tracker (SORT/DeepSORT) to avoid double counts across frames.

## Troubleshooting
- Model file not found: confirm `models/best.pt` exists and path is correct.
- No detections: verify model was trained for vehicle class and adjust `conf`.
- Video window not appearing (headless server): write output to file and disable imshow.
- VideoWriter errors: ensure frame size matches (width × height) and codec is available.

## Performance Tips
- Use GPU-enabled PyTorch/Ultralytics for real-time processing.
- Crop ROIs to reduce image area for inference.
- Lower `imgsz` for higher FPS; tune `conf` to balance precision/recall.
- Batch processing or multi-threaded read/infer/write pipeline for higher throughput.

## Future Enhancements
- Integrate Multi-Object Tracking (SORT / DeepSORT) to stabilize counts and extract trajectories.
- Use per-class counting and per-vehicle-type thresholds.
- Add speed estimation using frame-to-frame centroid displacement and calibration.
- Build a streaming/publishing pipeline (MQTT/HTTP) to push metrics to dashboards.
- Add unit/integration tests and CI for reproducible experiments.
- Export to ONNX / TensorRT for optimized cross-platform deployments.

## Contributing
- Fork the repository, create a feature branch, add tests/documentation, and submit a pull request.
- Include a clear description of changes and any dataset or model changes.
- Report issues through the project issue tracker and include reproducible steps.

## License
Add a LICENSE file to the repo (e.g., MIT) if you intend to open-source this project.

## Contact
For questions or collaboration, refer to the repository or the project owner (links in original project documentation).
