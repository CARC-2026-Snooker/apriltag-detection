import cv2
import numpy as np
import math
from pupil_apriltags import Detector
from collections import deque
import os
import sys
import time

# --- 1. DEFINE YOUR FIELD / KNOWN TAGS ---
KNOWN_TAGS = { 
    1: {"x": 1.262, "y": 1.05, "facing_angle": 270},
    2: {"x": -0.062, "y": 1.05, "facing_angle": 270},
    3: {"x": 0, "y": 0, "facing_angle": 270},
    4: {"x": 0.6, "y": 0, "facing_angle": 270},
    5: {"x": 1.2, "y": 0, "facing_angle": 270},
}

# Unique BGR colors assigned to each tag ID
TAG_COLORS = {
    1: (255, 105, 180), # Hot Pink
    2: (0, 255, 255),   # Yellow
    3: (0, 165, 255),   # Orange
    4: (255, 255, 0),   # Cyan
    5: (255, 0, 255),   # Magenta
}

# Format: (x_min, y_min, x_max, y_max) in meters.
ARENA_BOUNDS = (0.0, 0.0, 1.2, 3.6)

# --- 2. ADVANCED STABILITY & JITTER FILTERS ---
# MAX_TAG_DISTANCE = 2.2       # Cut out noisy long-range tags (ignores back tags when far away)
MAX_TAG_DISTANCE = 10       # Cut out noisy long-range tags (ignores back tags when far away)
OUT_OF_BOUNDS_LEEWAY = 0.15  # Disregard coordinates more than 15cm outside field walls
# MAX_POSE_JUMP = 0.20         # Reject single-frame jumps greater than 20cm from last tracked state
MAX_POSE_JUMP = 1.0         # Reject single-frame jumps greater than 20cm from last tracked state
TRACKING_TIMEOUT = 0.3       # Time in seconds before forgetting last position when blind (Fixes turn-away bug)

# --- 3. ROI (REGION OF INTEREST) OPTIMIZATION ---
# Adjust these values to cut out dead space (e.g., 0.25 = 25% cropped)
CROP_TOP_PERCENT = 0.20       # Cuts the top X% of the camera view (ceiling/sky)
CROP_BOTTOM_PERCENT = 0.20    # Cuts the bottom X% of the camera view (robot chassis/floor)
MASK_ALPHA = 0.3              # Transparency of the red ignore-mask (0.0 = invisible, 1.0 = solid)

SCALE = 200    # Pixels per meter.
PADDING = 60   # Pixel border around the arena

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
    tag_space_dx = R[2, 0]
    tag_space_dz = R[2, 2]
    
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
    
    # CRITICAL: Always measure from outer black edge to outer black edge with a ruler!
    tag_size_meters = 0.11 

    if sys.platform.startswith('win'):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise IOError("Webcam could not be opened.")
    
    # Enforce streaming compressed format for speed
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Set high-res configuration limits
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- NEW FIXES FOR "GELATIN/BOUNCY" VIDEO ---
    # 1. Force OpenCV to always grab the newest frame (eliminates tracking lag)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 2. Force the camera to maintain a high framerate to reduce motion blur
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 3. Disable Auto-Focus (if your laptop camera supports it) to stop lens "breathing"
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # --------------------------------------------

    for _ in range(5):
        cap.read()
        time.sleep(0.1)

    # DYNAMIC CALIBRATION SCALING: Let's read what resolution the camera actually opened at
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened successfully at: {actual_w}x{actual_h}")

    # CHOOSE YOUR ACTIVE CAMERA ENVIRONMENT HERE:
    # ----------------------------------------------------------------------
    # A: LAPTOP CAMERA CONFIG (Base parameters captured at 720p)
    base_params = [991.82, 990.41, 664.28, 375.20]
    scale_factor = actual_h / 720.0
    
    # B: EXTERNAL 4K CAMERA CONFIG (Uncomment this when swapping to the real camera!)
    # base_params = [2425.06, 2439.43, 997.08, 404.67] # (Captured at 1080p)
    # scale_factor = actual_h / 1080.0
    # ----------------------------------------------------------------------

    camera_params = [p * scale_factor for p in base_params]
    print(f"Scaled camera matrix applied: {camera_params}")

    history_raw_x = deque(maxlen=15)
    history_raw_y = deque(maxlen=15)

    print("Press 'q' to quit.")

    last_x, last_y = 0.0, 0.0
    last_dx, last_dy = 1.0, 0.0
    initialized = False
    last_seen_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # Calculate cropping boundaries
        y_start = int(h * CROP_TOP_PERCENT)
        y_end = int(h * (1.0 - CROP_BOTTOM_PERCENT))

        # Draw the visual red dead-zone mask
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, y_start), (0, 0, 255), -1)
        cv2.rectangle(overlay, (0, y_end), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, frame)

        # Draw a bright green line showing the active scanning area boundaries
        cv2.line(frame, (0, y_start), (w, y_start), (0, 255, 0), 2)
        cv2.line(frame, (0, y_end), (w, y_end), (0, 255, 0), 2)
        cv2.putText(frame, "ACTIVE SCANNING ZONE", (10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Slice the grayscale frame down to save CPU processing power
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_cropped = gray[y_start:y_end, :]

        # CRITICAL MATH FIX: Adjust the optical Y-center (cy) by the cropped amount
        adj_camera_params = [
            camera_params[0],          # fx
            camera_params[1],          # fy
            camera_params[2],          # cx
            camera_params[3] - y_start # cy shifted
        ]

        # Detect tags only inside the smaller cropped image
        tags = detector.detect(gray_cropped, estimate_tag_pose=True, camera_params=adj_camera_params, tag_size=tag_size_meters)
        debug_tag_estimates = []

        for tag in tags:
            tag_color = TAG_COLORS.get(tag.tag_id, (0, 255, 0))
            
            # Re-offset the visual drawing coordinates back to the global frame size
            tag.corners[:, 1] += y_start
            tag.center[1] += y_start

            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), tag_color, 2)
            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            t, R = tag.pose_t, tag.pose_R
            pose_data = get_global_camera_pose(tag.tag_id, t, R)
            
            if pose_data:
                dist = math.sqrt(t[0][0]**2 + t[1][0]**2 + t[2][0]**2)
                
                if dist > MAX_TAG_DISTANCE:
                    continue
                
                global_cam_x, global_cam_y, global_dx, global_dy = pose_data
                
                if not (ARENA_BOUNDS[0] - OUT_OF_BOUNDS_LEEWAY <= global_cam_x <= ARENA_BOUNDS[2] + OUT_OF_BOUNDS_LEEWAY and
                        ARENA_BOUNDS[1] - OUT_OF_BOUNDS_LEEWAY <= global_cam_y <= ARENA_BOUNDS[3] + OUT_OF_BOUNDS_LEEWAY):
                    continue
                
                if initialized:
                    jump_dist = math.sqrt((global_cam_x - last_x)**2 + (global_cam_y - last_y)**2)
                    if jump_dist > MAX_POSE_JUMP:
                        continue 
                
                debug_tag_estimates.append({
                    "id": tag.tag_id,
                    "pose": pose_data,
                    "distance": dist
                })

                text = f"ID:{tag.tag_id} -> Dist:{dist:.2f}m"
                cv2.putText(frame, text, (center[0] - 50, center[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)

        # Handle position averaging and continuous adaptive smoothing
        if len(debug_tag_estimates) > 0:
            last_seen_time = time.time()

            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            weighted_dx = 0.0
            weighted_dy = 0.0
            
            for entry in debug_tag_estimates:
                pose = entry["pose"]
                # DISTANCE WEIGHT: Closer tags have significantly higher reliability.
                weight = 1.0 / (entry["distance"]**3 + 0.001)
                
                weighted_x += pose[0] * weight
                weighted_y += pose[1] * weight
                weighted_dx += pose[2] * weight
                weighted_dy += pose[3] * weight
                total_weight += weight
                
            raw_avg_x = weighted_x / total_weight
            raw_avg_y = weighted_y / total_weight
            raw_avg_dx = weighted_dx / total_weight
            raw_avg_dy = weighted_dy / total_weight
            
            mag = math.sqrt(raw_avg_dx**2 + raw_avg_dy**2)
            if mag > 0:
                raw_avg_dx /= mag
                raw_avg_dy /= mag

            history_raw_x.append(raw_avg_x)
            history_raw_y.append(raw_avg_y)
            
            if not initialized:
                last_x, last_y = raw_avg_x, raw_avg_y
                last_dx, last_dy = raw_avg_dx, raw_avg_dy
                initialized = True
            
            if len(history_raw_x) >= 2:
                std_x = np.std(history_raw_x)
                std_y = np.std(history_raw_y)
                movement_metric = math.sqrt(std_x**2 + std_y**2)
            else:
                movement_metric = 0.0
                
            still_threshold = 0.003  
            move_threshold = 0.030   
            
            if movement_metric <= still_threshold:
                alpha = 0.01  
                state_text = f"STILL (Alpha: {alpha:.2f})"
                text_color = (0, 150, 0) 
            elif movement_metric >= move_threshold:
                alpha = 0.25  
                state_text = f"MOVING (Alpha: {alpha:.2f})"
                text_color = (0, 255, 255) 
            else:
                ratio = (movement_metric - still_threshold) / (move_threshold - still_threshold)
                alpha = 0.02 + ratio * (0.25 - 0.01)
                state_text = f"TRANSITIONING (Alpha: {alpha:.2f})"
                text_color = (0, 255, 0) 
                
            last_x = alpha * raw_avg_x + (1.0 - alpha) * last_x
            last_y = alpha * raw_avg_y + (1.0 - alpha) * last_y
            last_dx = alpha * raw_avg_dx + (1.0 - alpha) * last_dx
            last_dy = alpha * raw_avg_dy + (1.0 - alpha) * last_dy
            
            mag_smooth = math.sqrt(last_dx**2 + last_dy**2)
            if mag_smooth > 0:
                last_dx /= mag_smooth
                last_dy /= mag_smooth
                
            cv2.putText(frame, f"{state_text} X={last_x*100:.1f}cm Y={last_y*100:.1f}cm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        else:
            if initialized and (time.time() - last_seen_time > TRACKING_TIMEOUT):
                initialized = False  # Dropping initialization forces a free snap-reinitialization on next sight!
                history_raw_x.clear()
                history_raw_y.clear()

            cv2.putText(frame, f"LOST TAGS - LAST POS: X={last_x*100:.1f}cm Y={last_y*100:.1f}cm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        # --- DRAW THE 2D REAL-TIME MAP ---
        map_canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        
        # 1. Draw the customizable Arena Boundary Rectangle
        pt_top_left = world_to_pixel(ARENA_BOUNDS[0], ARENA_BOUNDS[1])
        pt_bottom_right = world_to_pixel(ARENA_BOUNDS[2], ARENA_BOUNDS[3])
        cv2.rectangle(map_canvas, pt_top_left, pt_bottom_right, (0, 150, 150), 2)
        
        # 2. Draw subtle grid lines strictly inside the arena bounds
        grid_spacing = 0.5 
        for grid_step in np.arange(ARENA_BOUNDS[0], ARENA_BOUNDS[2] + 0.1, grid_spacing):
            start = world_to_pixel(grid_step, ARENA_BOUNDS[1])
            end = world_to_pixel(grid_step, ARENA_BOUNDS[3])
            cv2.line(map_canvas, start, end, (40, 40, 40), 1)
            
        for grid_step in np.arange(ARENA_BOUNDS[1], ARENA_BOUNDS[3] + 0.1, grid_spacing):
            start = world_to_pixel(ARENA_BOUNDS[0], grid_step)
            end = world_to_pixel(ARENA_BOUNDS[2], grid_step)
            cv2.line(map_canvas, start, end, (40, 40, 40), 1)

        # Draw additional field markings
        draw_map_line(map_canvas, x1=0.0, y1=2.1, x2=1.2, y2=2.1, color=(255, 255, 255), thickness=2)
        draw_map_line(map_canvas, x1=0.0, y1=2.4, x2=1.2, y2=2.4, color=(0, 150, 0), thickness=2)
        draw_map_line(map_canvas, x1=0.0, y1=2.5, x2=1.2, y2=2.5, color=(150, 150, 150), thickness=2)
            
        # 3. Plot Known AprilTags (Updated to match individual identity colors)
        for tag_id, info in KNOWN_TAGS.items():
            pt = world_to_pixel(info["x"], info["y"])
            color = TAG_COLORS.get(tag_id, (0, 0, 255))
            
            cv2.rectangle(map_canvas, (pt[0]-6, pt[1]-6), (pt[0]+6, pt[1]+6), color, -1)
            cv2.putText(map_canvas, f"ID {tag_id}", (pt[0]-14, pt[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            rad = math.radians(info["facing_angle"])
            arrow_end = world_to_pixel(info["x"] - 0.12 * math.cos(rad), info["y"] - 0.12 * math.sin(rad))
            cv2.arrowedLine(map_canvas, pt, arrow_end, color, 1, tipLength=0.3)

        # 4. Plot Debug Tracking Lines & Individual Camera Heading Vectors
        for entry in debug_tag_estimates:
            tag_id = entry["id"]
            pose = entry["pose"]
            
            if tag_id in KNOWN_TAGS:
                tag_info = KNOWN_TAGS[tag_id]
                pt_tag_real = world_to_pixel(tag_info["x"], tag_info["y"])
                pt_cam_pred = world_to_pixel(pose[0], pose[1])
                color = TAG_COLORS.get(tag_id, (255, 255, 255))
                
                cv2.line(map_canvas, pt_tag_real, pt_cam_pred, color, 1)
                cv2.circle(map_canvas, pt_cam_pred, 3, color, -1)
                
                arrow_length = 0.15 
                pt_arrow_end = world_to_pixel(pose[0] + pose[2] * arrow_length, pose[1] + pose[3] * arrow_length)
                cv2.arrowedLine(map_canvas, pt_cam_pred, pt_arrow_end, color, 1, tipLength=0.25)
         
        # 5. Plot Blended Smoothed Robot Position and Orientation Vector
        if initialized:
            cam_pt = world_to_pixel(last_x, last_y)
            cv2.circle(map_canvas, cam_pt, 10, (255, 80, 80), -1)
            
            heading_end = world_to_pixel(last_x + last_dx * 0.25, last_y + last_dy * 0.25)
            cv2.arrowedLine(map_canvas, cam_pt, heading_end, (0, 255, 0), 2, tipLength=0.2)
            cv2.putText(map_canvas, "Robot", (cam_pt[0]-16, cam_pt[1]+24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Display Windows
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