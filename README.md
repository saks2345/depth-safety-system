# AI-Based Monocular Depth Estimation System

Low-cost 3D perception pipeline using a single webcam. Estimates depth in real time, detects objects, and triggers an automatic motor cutoff via Arduino when something enters the safety zone.

*Google Developers Hacktonic 2.0 – Finalist*

## Features
- **Depth Anything V2 (Base)** monocular depth estimation (CPU, runs every 4th frame for speed)
- **Tilt correction**: least-squares plane fit on the bottom third of the depth map, subtracted to flatten ground/camera tilt
- **YOLOv8s** object detection with per-box "closeness" (95th-percentile depth in the box)
- **3D safety-bubble logic**: EMA smoothing + hysteresis, adjustable danger threshold
- **Arduino integration** over serial: `G` (go) / `S` (stop) for motor cutoff
- **Multi-modal alerts**: on-screen status, bounding-box colours, audible beep, motor stop
- 2x2 live dashboard: video + YOLO, raw depth, tilt-corrected depth, status panel

## Setup
```bash
pip install -r requirements.txt
```
YOLO weights (`yolov8s.pt`) and the Depth Anything model download automatically on first run.

Edit `self.SERIAL_PORT` in `zeo.py` (default `COM5`) to match your Arduino. If no board is found, it runs in simulation mode without motor control.

## Run
```bash
python zeo.py
```
Keys: `Q`/`Esc` quit · `+`/`-` change danger threshold · `C` print status

`tools/find_camera.py` lists available camera indexes.

## Arduino
The Arduino listens for single characters over serial at 9600 baud: `G` = motors on, `S` = motors off.
