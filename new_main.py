import cv2
import numpy as np
import math
from pupil_apriltags import Detector
from collections import deque
import os
import sys
import time
# --- 1. DEFINE YOUR FIELD / KNOWN TAGS ---
# Dictionary structure: { tag_id: {"x": global_X_meters, "y": global_Y_meters, "facing_angle": degrees} }
KNOWN_TAGS = { 
    1: {"x": 1.262, "y": 1.05, "facing_angle": 270},
    2: {"x": -0.062, "y": 1.05, "facing_angle": 270},
    3: {"x": 0, "y": 0, "facing_angle": 270},
    4: {"x": 0.6, "y": 0, "facing_angle": 270},
    5: {"x": 1.2, "y": 0, "facing_angle": 270},
}

TAG_COLORS = {
    1: (255, 105, 180), # Hot Pink
    2: (0, 255, 255),   # Yellow
    3: (0, 165, 255),   # Orange
    4: (255, 255, 0),   # Cyan
    5: (255, 0, 255),   # Magenta
}

# Format: (x_min, y_min, x_max, y_max) in meters.
ARENA_BOUNDS = (0.0, 0.0, 1.2, 3.6)

SCALE = 220    # Pixels per meter. Increase to make the window larger.
PADDING = 60   # Pixel border around the arena so tags/text don't hit the screen edge

# Calculate the precise pixel size needed for the window
arena_w_meters = ARENA_BOUNDS[2] - ARENA_BOUNDS[0]
arena_h_meters = ARENA_BOUNDS[3] - ARENA_BOUNDS[1]

CANVAS_WIDTH = int(arena_w_meters * SCALE) + (2 * PADDING)
CANVAS_HEIGHT = int(arena_h_meters * SCALE) + (2 * PADDING)

def get_global_camera_pose(tag_id, t, R):
    """Calculates the camera's global X, Y position and orientation vector from a single tag."""
    if tag_id not in KNOWN_TAGS:
        return None

    # 1. Calculate camera position relative to the tag
    cam_pos_in_tag_frame = -np.transpose(R) @ t
    local_x = cam_pos_in_tag_frame[0][0]
    local_z = cam_pos_in_tag_frame[2][0]

    # 2. Map this local position to the global map using the tag's known position
    tag_info = KNOWN_TAGS[tag_id]
    global_tag_x = tag_info["x"]
    global_tag_y = tag_info["y"]
    
    theta = math.radians(tag_info["facing_angle"])
    
    global_cam_x = global_tag_x + (local_z * math.cos(theta)) - (local_x * math.sin(theta))
    global_cam_y = global_tag_y + (local_z * math.sin(theta)) + (local_x * math.cos(theta))
    
    # 3. Calculate camera orientation vector in global space
    # Camera's local forward vector is Z [0, 0, 1]. In tag space, this is the 3rd column of R.T (or 3rd row of R)
    tag_space_dx = R[2, 0]
    tag_space_dz = R[2, 2]
    
    # Rotate tag-space direction vector into global space coordinates
    global_dx = (tag_space_dz * math.cos(theta)) - (tag_space_dx * math.sin(theta))
    global_dy = (tag_space_dz * math.sin(theta)) + (tag_space_dx * math.cos(theta))
    
    return (global_cam_x, global_cam_y, global_dx, global_dy)

def world_to_pixel(x, y):
    """Maps global meters to pixel positions based on current arena dimensions."""
    px = int((x - ARENA_BOUNDS[0]) * SCALE) + PADDING
    py = int((y - ARENA_BOUNDS[1]) * SCALE) + PADDING
    return (px, py)

def draw_map_line(canvas, x1, y1, x2, y2, color=(255, 255, 255), thickness=2):
    """Draws a line on the canvas from real-world coordinate A to B."""
    pt_a = world_to_pixel(x1, y1)
    pt_b = world_to_pixel(x2, y2)
    cv2.line(canvas, pt_a, pt_b, color, thickness)

def main():
    detector = Detector(families="tag36h11")
    tag_size_meters = 0.11

    # Define resolution targets
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720

    # Automatically scale your 1080p calibration matrix to match the target resolution
    scale_factor = TARGET_HEIGHT / 1080.0
    base_params = [991.82, 990.41, 664.28, 375.20] # for laptop camera
    # base_params = [2425.06, 2439.43, 997.08, 404.67] # for actual camera
    camera_params = [p * scale_factor for p in base_params]

    # Keeping a short history buffer of raw detections to measure real-time noise/movement variance
    history_raw_x = deque(maxlen=15)
    history_raw_y = deque(maxlen=15)

    if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise IOError("Webcam could not be opened.")
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    print("Press 'q' to quit.")

    # State variables
    last_x, last_y = 0.0, 0.0
    last_dx, last_dy = 1.0, 0.0
    initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, -1) # for actual camera

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size_meters)

        global_estimates = []
        debug_tag_estimates = []

        for tag in tags:
            # Draw original bounding box overlays
            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            # Get 3D pose matrix configurations
            t, R = tag.pose_t, tag.pose_R
            pose_data = get_global_camera_pose(tag.tag_id, t, R)
            
            if pose_data:
                global_estimates.append(pose_data)

                debug_tag_estimates.append((tag.tag_id, pose_data))

                text = f"ID:{tag.tag_id} -> X:{pose_data[0]:.2f} Y:{pose_data[1]:.2f}"
                cv2.putText(frame, text, (center[0] - 50, center[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Handle position averaging and continuous adaptive smoothing
        if len(global_estimates) > 0:
            raw_avg_x = sum(p[0] for p in global_estimates) / len(global_estimates)
            raw_avg_y = sum(p[1] for p in global_estimates) / len(global_estimates)
            raw_avg_dx = sum(p[2] for p in global_estimates) / len(global_estimates)
            raw_avg_dy = sum(p[3] for p in global_estimates) / len(global_estimates)
            
            # Normalize directional heading components
            mag = math.sqrt(raw_avg_dx**2 + raw_avg_dy**2)
            if mag > 0:
                raw_avg_dx /= mag
                raw_avg_dy /= mag

            # Append newest snapshot to metric buffer
            history_raw_x.append(raw_avg_x)
            history_raw_y.append(raw_avg_y)
            
            if not initialized:
                last_x, last_y = raw_avg_x, raw_avg_y
                last_dx, last_dy = raw_avg_dx, raw_avg_dy
                initialized = True
            
            # Calculate standard deviation over the short 15-frame history window
            if len(history_raw_x) >= 2:
                std_x = np.std(history_raw_x)
                std_y = np.std(history_raw_y)
                movement_metric = math.sqrt(std_x**2 + std_y**2)
            else:
                movement_metric = 0.0
                
            # Define limits for continuous mapping transition bounds (meters)
            still_threshold = 0.003  # Variance below 3mm -> perfectly still
            move_threshold = 0.030   # Variance above 3cm -> clearly moving fast
            
            if movement_metric <= still_threshold:
                # alpha = 0.02  # Maximum low-pass filter smoothing to eliminate coordinate flicker
                alpha = 0.01  # Maximum low-pass filter smoothing to eliminate coordinate flicker
                state_text = f"STILL (Alpha: {alpha:.2f})"
                text_color = (0, 150, 0) # Dark Green
            elif movement_metric >= move_threshold:
                alpha = 0.30  # Low smoothing factor, updates immediately with no lag
                state_text = f"MOVING (Alpha: {alpha:.2f})"
                text_color = (0, 255, 255) # Yellow
            else:
                # Continuous interpolation between still and moving states!
                ratio = (movement_metric - still_threshold) / (move_threshold - still_threshold)
                alpha = 0.02 + ratio * (0.30 - 0.02)
                state_text = f"TRANSITIONING (Alpha: {alpha:.2f})"
                text_color = (0, 255, 0) # Light Green
                
            # Apply Adaptive Exponential Moving Average (EMA) equations
            last_x = alpha * raw_avg_x + (1.0 - alpha) * last_x
            last_y = alpha * raw_avg_y + (1.0 - alpha) * last_y
            last_dx = alpha * raw_avg_dx + (1.0 - alpha) * last_dx
            last_dy = alpha * raw_avg_dy + (1.0 - alpha) * last_dy
            
            # Normalize final smoothed vectors
            mag_smooth = math.sqrt(last_dx**2 + last_dy**2)
            if mag_smooth > 0:
                last_dx /= mag_smooth
                last_dy /= mag_smooth
                
            cv2.putText(frame, f"{state_text} X={last_x*100:.1f}cm Y={last_y*100:.1f}cm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        else:
            cv2.putText(frame, f"LOST TAGS - LAST POS: X={last_x*100:.1f}cm Y={last_y*100:.1f}cm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # --- DRAW THE 2D REAL-TIME MAP ---
        map_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        
        # 1. Draw the customizable Arena Boundary Rectangle
        pt_top_left = world_to_pixel(ARENA_BOUNDS[0], ARENA_BOUNDS[1])
        pt_bottom_right = world_to_pixel(ARENA_BOUNDS[2], ARENA_BOUNDS[3])
        # Draws a dark yellow rectangle outlining the field
        cv2.rectangle(map_canvas, pt_top_left, pt_bottom_right, (0, 150, 150), 2)
        
        # 2. Draw subtle grid lines strictly inside the arena bounds
        grid_spacing = 0.5 # 0.5 meters
        for grid_step in np.arange(ARENA_BOUNDS[0], ARENA_BOUNDS[2] + 0.1, grid_spacing):
            start = world_to_pixel(grid_step, ARENA_BOUNDS[1])
            end = world_to_pixel(grid_step, ARENA_BOUNDS[3])
            cv2.line(map_canvas, start, end, (40, 40, 40), 1)
            
        for grid_step in np.arange(ARENA_BOUNDS[1], ARENA_BOUNDS[3] + 0.1, grid_spacing):
            start = world_to_pixel(ARENA_BOUNDS[0], grid_step)
            end = world_to_pixel(ARENA_BOUNDS[2], grid_step)
            cv2.line(map_canvas, start, end, (40, 40, 40), 1)

        # draw additional line markings
        draw_map_line(map_canvas, x1=0.0, y1=2.1, x2=1.2, y2=2.1, color=(255, 255, 255), thickness=2)
        draw_map_line(map_canvas, x1=0.0, y1=2.4, x2=1.2, y2=2.4, color=(0, 150, 0), thickness=2)
        draw_map_line(map_canvas, x1=0.0, y1=2.5, x2=1.2, y2=2.5, color=(150, 150, 150), thickness=2)
            
        # Plot Known AprilTags
        for tag_id, info in KNOWN_TAGS.items():
            pt = world_to_pixel(info["x"], info["y"])
            cv2.rectangle(map_canvas, (pt[0]-6, pt[1]-6), (pt[0]+6, pt[1]+6), (0, 0, 255), -1)
            cv2.putText(map_canvas, f"ID {tag_id}", (pt[0]-14, pt[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Heading directional line for tags
            rad = math.radians(info["facing_angle"])
            
            # ---> CHANGE THIS LINE <---
            # Change the '+' signs to '-' signs to flip the visual arrow 180 degrees!
            arrow_end = world_to_pixel(info["x"] - 0.12 * math.cos(rad), info["y"] - 0.12 * math.sin(rad))
            
            cv2.arrowedLine(map_canvas, pt, arrow_end, (255, 100, 100), 1, tipLength=0.3)

        # Plot Debug Tracking Lines (Tag to individual estimated camera pos)
        for tag_id, pose in debug_tag_estimates:
            if tag_id in KNOWN_TAGS:
                tag_info = KNOWN_TAGS[tag_id]
                
                # A: The real physical location of the tag on the wall
                pt_tag_real = world_to_pixel(tag_info["x"], tag_info["y"])
                
                # B: The location of the camera, AS PREDICTED by this specific tag
                pt_cam_pred = world_to_pixel(pose[0], pose[1])
                
                # Get the unique color for this tag (default to white if ID not in dict)
                color = TAG_COLORS.get(tag_id, (255, 255, 255))
                
                # Draw a thin line connecting them
                cv2.line(map_canvas, pt_tag_real, pt_cam_pred, color, 1)
                
                # Draw a small dot at the predicted camera coordinate
                cv2.circle(map_canvas, pt_cam_pred, 3, color, -1)
         
        # Plot Robot Position and Orientation Vector
        if initialized:
            cam_pt = world_to_pixel(last_x, last_y)
            # Draw Robot Chassis
            cv2.circle(map_canvas, cam_pt, 10, (255, 80, 80), -1)
            # Draw Heading Angle Vector (Length scaled to 25cm in map units)
            heading_end = world_to_pixel(last_x + last_dx * 0.25, last_y + last_dy * 0.25)
            cv2.arrowedLine(map_canvas, cam_pt, heading_end, (0, 255, 0), 2, tipLength=0.2)
            cv2.putText(map_canvas, "Robot", (cam_pt[0]-16, cam_pt[1]+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Show windows side by side
        cv2.namedWindow("AprilTag Localization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AprilTag Localization", 800, 450)
        cv2.imshow("AprilTag Localization", frame)
        
        cv2.imshow("2D Real-Time Field Map", map_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()