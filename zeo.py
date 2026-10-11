## ⚙️ Fully Optimized and Enhanced Code: TEST_SERIAL_GO_STOP.py (2x2 Grid)

# Depth Anything + YOLO + Motor Speed Simulation + Bounding Boxes
# - INTEGRATED with ARDUINO for GO ('G') / STOP ('S') control.

import cv2
import time
import numpy as np
from PIL import Image

# --- NEW: Import Serial Library ---
try:
    import serial 
    import serial.tools.list_ports
except ImportError:
    print("ERROR: Missing pyserial dependency.")
    print("Please ensure you run: pip install pyserial")
    exit()

# Beep alert
try:
    import winsound
    # Reduced beep duration for less interruption
    def beep_alert(): winsound.Beep(1000, 150)
except:
    def beep_alert(): print("[BEEP]")

# --- Dependency Check ---
try:
    from transformers import pipeline
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Missing dependencies.")
    print("Please ensure you run: pip install ultralytics transformers Pillow")
    exit()

class DepthAnythingYOLOMotorBoxes:
    def __init__(self, cam_index=0):
        print("=" * 70)
        print("  Depth Anything V2 + YOLO + Motor Speed Simulation (Max Accuracy)")
        print("=" * 70)

        # --- Serial Setup ---
        # IMPORTANT: Change 'COM3' to your Arduino's serial port (e.g., '/dev/ttyACM0' on Linux, 'COM3' on Windows)
        self.SERIAL_PORT = 'COM5' 
        self.BAUD_RATE = 9600
        self.ser = None 
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=0.1)
            time.sleep(2) # Wait for Arduino to reset
            print(f"✓ Serial port {self.SERIAL_PORT} opened successfully.")
        except serial.SerialException as e:
            print(f"❌ ERROR opening serial port {self.SERIAL_PORT}: {e}")
            print("Running in simulation mode (no motor control).")
            # Continue without motor control if serial fails
        # --------------------

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Cannot open camera at index {cam_index}!")
        # Use more standard resolutions for better compatibility
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Setting FPS might not always work, but good practice
        self.cap.set(cv2.CAP_PROP_FPS, 30) 
        print(f"✓ Camera initialized (index={cam_index})")

        # --- Depth Model ---
        # IMPROVEMENT 1: Using the more accurate 'Base' model (slower, but better quality)
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=-1 # CPU mode
        )
        print("✓ Depth Anything V2 Base loaded (Will run every 4th frame)")

        # --- YOLOv8 ---
        # IMPROVEMENT 2: Using the more capable 's' (Small) model for better detection
        self.yolo = YOLO("yolov8s.pt")
        print("✓ YOLOv8 Small loaded (Optimized input size)")

        # --- Parameters ---
        # Danger threshold set for earlier detection
        self.danger_threshold = 0.782
        self.min_confidence = 0.5
        # Reduced EMA alpha for more stable/persistent closeness reading
        self.ema_alpha = 0.3 
        self.ema_closeness = None
        
        # PERFORMANCE FIX: Run depth estimation only every 4th frame
        self.frame_skip = 4 
        self.frame_counter = 0

        # --- Motor Simulation ---
        self.motor_speed = 100
        self.min_motor_speed = 20 # Minimum speed the code can drop to
        self.max_motor_speed = 100

        # --- Stability (Hysteresis) ---
        self.danger_consecutive_frames = 3
        # Increased required safe frames (0.5 sec) to keep danger alert persistent
        self.safe_consecutive_frames = 15
        self.danger_frame_count = 0
        self.safe_frame_count = 0
        self.in_danger = False

        # --- FPS Tracking ---
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0

        print("\nSystem ready! Starting in 3 seconds...\n")
        time.sleep(3)
    
    # -------------------------------------------------
    # --- NEW/MODIFIED: Serial Sending Method (G/S Command) ---
    def send_motor_command_to_arduino(self):
        if self.ser and self.ser.is_open:
            # Logic: If the motor speed is at or near the minimum (20%), 
            # we consider it a DANGER/STOP signal. Otherwise, it's SAFE/GO.
            
            # Using a threshold just above min_motor_speed to avoid floating point issues.
            STOP_THRESHOLD = self.min_motor_speed + 1 

            command = 'S'  # Default to Stop
            if self.motor_speed > STOP_THRESHOLD:
                command = 'G'
            
            try:
                # Send the single character command (ASCII encoded)
                # No newline needed for single char commands
                self.ser.write(command.encode('utf-8'))
            except serial.SerialTimeoutException:
                print("⚠ Serial write timeout.")
            except Exception as e:
                # This handles disconnection errors
                # print(f"⚠ Serial write error: {e}")
                pass
    # -------------------------------------------------

    def tilt_correct(self, depth_map):
        """
        Robust software-only tilt flattening via plane fitting on the bottom third of the image.
        (Original logic unchanged)
        """
        h, w = depth_map.shape
        # Define the region for plane fitting (bottom 1/3 of the image)
        fit_region_start_y = int(h * 2 / 3)
        
        if fit_region_start_y >= h - 1 or h < 3: 
             return depth_map

        y_fit, x_fit = np.mgrid[fit_region_start_y:h, 0:w]
        depth_fit_region = depth_map[fit_region_start_y:h, 0:w]

        # Prepare A matrix for least squares (x, y, 1)
        X = np.stack((x_fit.ravel(), y_fit.ravel(), np.ones_like(x_fit).ravel()), axis=1).astype(float)
        Y = depth_fit_region.ravel().astype(float)

        # Calculate plane coefficients
        try:
            if X.shape[0] < 3: 
                return depth_map
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            return depth_map 
        
        # Reconstruct the plane across the ENTIRE image
        y_full, x_full = np.mgrid[0:h, 0:w]
        plane = (coeffs[0] * x_full.astype(float) + coeffs[1] * y_full.astype(float) + coeffs[2])
        
        # Subtract the calculated plane from the full depth map
        corrected = depth_map.astype(float) - plane
        
        # Re-normalize the corrected map to 0-1 range (1.0 is closest)
        c_min = np.min(corrected)
        c_max = np.max(corrected)
        
        if (c_max - c_min) > 1e-6:
              corrected = (corrected - c_min) / (c_max - c_min)
        else:
              return np.clip(corrected - c_min, 0, 1)

        return np.clip(corrected, 0, 1)

    # -------------------------------------------------
    def estimate_depth(self, frame):
        """Depth estimation with normalization and TILT CORRECTION (Original logic unchanged)"""
        small_frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Depth Inference
        result = self.depth_pipe(pil_image)
        depth_map = np.array(result["depth"])
        
        # Normalize (0 to 1, 1.0 is closest)
        dmin, dmax = np.percentile(depth_map, 5), np.percentile(depth_map, 95)
        if dmax - dmin < 1e-6:
              depth_norm = np.zeros_like(depth_map)
        else:
              depth_norm = np.clip((depth_map - dmin) / (dmax - dmin + 1e-6), 0, 1)
        
        depth_norm_tilted = 1.0 - depth_norm # TILTED normalized map

        # 1. Corrected Map (Used for closeness calculation)
        depth_norm_corrected = self.tilt_correct(depth_norm_tilted.copy())
        
        # Resize all normalized maps back up to frame size
        h, w, _ = frame.shape
        depth_norm_final_corrected = cv2.resize(depth_norm_corrected, (w, h), interpolation=cv2.INTER_LINEAR)
        depth_norm_final_tilted = cv2.resize(depth_norm_tilted, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Color Visualization of Tilted Map
        depth_vis_tilted = (depth_norm_final_tilted * 255).astype(np.uint8)
        depth_color_tilted = cv2.applyColorMap(depth_vis_tilted, cv2.COLORMAP_INFERNO)
        
        # 3. Color Visualization of Corrected Map
        depth_vis_corrected = (depth_norm_final_corrected * 255).astype(np.uint8)
        depth_color_corrected = cv2.applyColorMap(depth_vis_corrected, cv2.COLORMAP_INFERNO)
        
        return depth_norm_final_corrected, depth_color_tilted, depth_color_corrected

    # -------------------------------------------------
    def get_box_closeness(self, depth_norm, box):
        """Compute closeness of object using depth map region (95th percentile) (Original logic unchanged)"""
        x1, y1, x2, y2 = map(int, box)
        
        h, w = depth_norm.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        region = depth_norm[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        # IMPROVEMENT 3: Use 95th percentile to estimate the absolute closest point
        return float(np.percentile(region, 95)) 

    # -------------------------------------------------
    def update_motor_speed(self, closeness):
        """Simulate motor speed adjustment (Original logic unchanged)"""
        if closeness > self.danger_threshold:
            self.motor_speed = max(self.min_motor_speed, self.motor_speed - 5)
        else:
            self.motor_speed = min(self.max_motor_speed, self.motor_speed + 3)
        return self.motor_speed

    # -------------------------------------------------
    def run(self):
        cv2.namedWindow("Real-Time Depth & Safety System", cv2.WINDOW_NORMAL)
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize with black/zeros
        depth_norm_corrected = np.zeros((h, w)) 
        depth_color_tilted = np.zeros((h, w, 3), dtype=np.uint8) 
        depth_color_corrected = np.zeros((h, w, 3), dtype=np.uint8) 

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠ Could not read frame from camera! Exiting.")
                break
            
            # --- PERFORMANCE FIX: Frame Skipping Logic ---
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip == 0:
                depth_norm_corrected, depth_color_tilted, depth_color_corrected = self.estimate_depth(frame)
                
            # --- YOLO detection ---
            # IMPROVEMENT 4: Increased input size to 480 for better YOLO accuracy
            results = self.yolo(frame, imgsz=480, verbose=False)[0] 
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            # --- Compute max closeness (using the CORRECTED map) ---
            max_closeness = 0
            vis_frame = frame.copy()
            
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    if confs[i] < self.min_confidence:
                        continue
                    
                    if depth_norm_corrected.size == 0: 
                        closeness = 0.0
                    else:
                        closeness = self.get_box_closeness(depth_norm_corrected, box)
                        
                    max_closeness = max(max_closeness, closeness)

                    # Draw bounding box on the REAL VIDEO frame
                    danger = closeness > self.danger_threshold
                    color = (0, 0, 255) if danger else (0, 255, 0)
                    x1, y1, x2, y2 = map(int, box) 
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"C:{closeness:.2f}"
                    cv2.putText(vis_frame, label, (x1, y1 - 5),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # --- EMA smoothing ---
            if self.ema_closeness is None:
                self.ema_closeness = max_closeness
            else:
                self.ema_closeness = (self.ema_alpha * max_closeness +
                                      (1 - self.ema_alpha) * self.ema_closeness)

            # --- Motor logic & Hysteresis ---
            self.update_motor_speed(self.ema_closeness)

            # --- SERIAL COMMAND SENDING ---
            self.send_motor_command_to_arduino()
            # ------------------------------

            danger = self.ema_closeness > self.danger_threshold

            if danger:
                self.danger_frame_count += 1
                self.safe_frame_count = 0
                if self.danger_frame_count >= self.danger_consecutive_frames and not self.in_danger:
                    self.in_danger = True
                    print(f"⚠ DANGER! Closeness={self.ema_closeness:.2f}, Motor={self.motor_speed:.0f}% (Sent 'S')")
                    beep_alert()
            else:
                self.safe_frame_count += 1
                self.danger_frame_count = 0
                if self.safe_frame_count >= self.safe_consecutive_frames and self.in_danger:
                    self.in_danger = False
                    print(f"✓ SAFE again - Motor={self.motor_speed:.0f}% (Sent 'G')")

            # --- Grid Composition & Info ---
            # --- Grid Composition & Info (Updated Logic) ---
            if self.ema_closeness > self.danger_threshold:
                status_text = "DANGER (Too Close!)"
                status_color = (0, 0, 255)
            else:
                status_text = "SAFE (Proceed)"
                status_color = (0, 255, 0)

            
            # --- Resize frames for the 2x2 grid (Half Width, Half Height) ---
            disp_w, disp_h = w // 2, h // 2 
            
            # Ensure depth maps are 3-channel for stacking with video frames
            vis_frame_resized = cv2.resize(vis_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            
            # Use original frame size for depth maps, they were already resized inside estimate_depth
            depth_tilted_quadrant = cv2.resize(depth_color_tilted, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            depth_corrected_quadrant = cv2.resize(depth_color_corrected, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # --- Control Panel/Status Quadrant (Bottom Right) ---
            info_quadrant = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)
            
            # FPS counter
            self.frame_count += 1
            if time.time() - self.last_time >= 1:
                self.fps = self.frame_count / (time.time() - self.last_time)
                self.frame_count = 0
                self.last_time = time.time()
                
            # Draw text on the info quadrant
            cv2.putText(info_quadrant, "CONTROL/STATUS", (10, 30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(info_quadrant, f"STATUS: {status_text}", (10, 70), font, 0.7, status_color, 2)
            cv2.putText(info_quadrant, f"Motor Speed: {self.motor_speed:.0f}%", (10, 110), font, 0.6, (0, 255, 255), 1)
            cv2.putText(info_quadrant, f"Closeness: {self.ema_closeness:.3f}", (10, 140), font, 0.6, text_color, 1)
            cv2.putText(info_quadrant, f"Threshold: {self.danger_threshold:.2f}", (10, 170), font, 0.6, (200, 200, 200), 1)
            cv2.putText(info_quadrant, f"FPS: {self.fps:.1f} (1/{self.frame_skip} Depth)", (10, disp_h - 20), font, 0.5, (255, 255, 0), 1)

            # --- Compose Grid ---
            # Top Row
            top_row = np.hstack([vis_frame_resized, depth_tilted_quadrant])
            # Bottom Row
            bottom_row = np.hstack([depth_corrected_quadrant, info_quadrant])
            
            # Final Display
            final_display = np.vstack([top_row, bottom_row])

            # Write titles over the quadrants
            cv2.putText(final_display, "1. REAL VIDEO (YOLO)", (10, 20), font, 0.5, (0, 255, 0), 1)
            cv2.putText(final_display, "2. DEPTH MAP (TILTED)", (disp_w + 10, 20), font, 0.5, (0, 255, 255), 1)
            cv2.putText(final_display, "3. DEPTH MAP (TILT CORRECTED)", (10, disp_h + 20), font, 0.5, (0, 255, 255), 1)

            # --- Show frame ---
            cv2.imshow("Real-Time Depth & Safety System", final_display)

            # --- Key controls ---
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key in [ord('+'), ord('=')]:
                self.danger_threshold = min(0.90, self.danger_threshold + 0.02)
                print(f"Threshold increased to {self.danger_threshold:.2f}")
            elif key in [ord('-'), ord('_')]:
                self.danger_threshold = max(0.60, self.danger_threshold - 0.02)
                print(f"Threshold decreased to {self.danger_threshold:.2f}")
            elif key == ord('c'):
                print(f"📊 Closeness={self.ema_closeness:.2f}, Motor={self.motor_speed:.0f}%")

        self.cap.release()
        cv2.destroyAllWindows()
        # --- NEW: Close Serial Port ---
        if self.ser and self.ser.is_open:
            # Send a final 'S' command to ensure the motor stops when the script exits
            try:
                self.ser.write('S'.encode('utf-8'))
            except:
                pass
            self.ser.close()
            print("✓ Serial connection closed.")
        # ------------------------------
        print("✓ Exited cleanly")


if __name__ == "__main__":
    try:
        print("Starting dependency checks...")
        # (Simplified dependency check)
        if 'pipeline' not in locals() or 'YOLO' not in locals():
            print("ERROR: Core dependencies failed to load.")
        else:
            # IMPORTANT: Before running, confirm your Arduino's serial port (e.g., 'COM3' or '/dev/ttyACM0') 
            # and update the self.SERIAL_PORT variable in the __init__ method.
            system = DepthAnythingYOLOMotorBoxes(cam_index=0) 
            system.run()
    except Exception as e:
        print(f"❌ An error occurred during startup or execution: {e}")