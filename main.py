import cv2
import numpy as np
import math
from pupil_apriltags import Detector
from collections import deque
from itertools import islice

# --- 1. DEFINE YOUR FIELD / KNOWN TAGS ---
# Dictionary structure: { tag_id: {"x": global_X_meters, "y": global_Y_meters, "facing_angle": degrees} }
# facing_angle: The direction the tag's Z-axis (pointing straight out of the tag) faces on your global map.
# e.g., If Tag 1 is on the left wall facing right (positive X direction), angle is 0.
KNOWN_TAGS = { 
    # test positions, not real one's yet
    # 1: 0.45 + 0.062 | 2: 0.45 + 0.124 + 0.244 + 0.062 
    # 3: 0.45 + 0.124 + 0.244 + 0.124 + 0.193 + 0.062 
    # 4: 0.45 + 0.124 + 0.244 + 0.124 + 0.193 + 0.124 + 0.188 + 0.062 
    1: {"x": 0.512, "y": 0, "facing_angle": 270},
    2: {"x": 0.88, "y": 0, "facing_angle": 270},
    3: {"x": 1.197, "y": 0, "facing_angle": 270},
    4: {"x": 1.509, "y": 0, "facing_angle": 270},
    5: {"x": 0, "y": 0, "facing_angle": 270},
    # Add all your competition tags here!
}

def get_global_camera_pose(tag_id, t, R):
    """Calculates the camera's global X, Y position from a single tag."""
    if tag_id not in KNOWN_TAGS:
        return None

    # 1. Calculate camera position relative to the tag
    # R.T is the transpose (inverse) of the rotation matrix
    cam_pos_in_tag_frame = -np.transpose(R) @ t
    
    # In the tag's local frame:
    # cam_pos_in_tag_frame[0][0] = Distance to the right (+) / left (-) of the tag center
    # cam_pos_in_tag_frame[1][0] = Distance below (+) / above (-) the tag center
    # cam_pos_in_tag_frame[2][0] = Distance straight out directly in front of the tag
    
    local_x = cam_pos_in_tag_frame[0][0]
    local_z = cam_pos_in_tag_frame[2][0]

    # 2. Map this local position to the global map using the tag's known position and rotation
    tag_info = KNOWN_TAGS[tag_id]
    global_tag_x = tag_info["x"]
    global_tag_y = tag_info["y"]
    
    # Convert tag's facing angle to radians
    theta = math.radians(tag_info["facing_angle"])
    
    # Standard 2D rotation matrix math to map local offset to global coordinates
    global_cam_x = global_tag_x + (local_z * math.cos(theta)) - (local_x * math.sin(theta))
    global_cam_y = global_tag_y + (local_z * math.sin(theta)) + (local_x * math.cos(theta))
    
    return (global_cam_x, global_cam_y)

def main():
    detector = Detector(families="tag36h11")
    tag_size_meters = 0.124
    camera_params = [629.54, 626.65, 302.72, 231.45]

    history_x = deque(maxlen=100) 
    history_y = deque(maxlen=100)
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to quit.")

    last_x = 0
    last_y = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(
            gray, 
            estimate_tag_pose=True, 
            camera_params=camera_params, 
            tag_size=tag_size_meters
        )

        global_estimates = [] # Store all calculated positions for this frame

        for tag in tags:
            # Draw bounding box
            corners = tag.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            t = tag.pose_t
            R = tag.pose_R

            # Calculate global position based on THIS specific tag
            estimated_pos = get_global_camera_pose(tag.tag_id, t, R)
            
            if estimated_pos:
                global_estimates.append(estimated_pos)
                text = f"ID:{tag.tag_id} -> CamX:{estimated_pos[0]:.2f} Y:{estimated_pos[1]:.2f}"
                cv2.putText(frame, text, (center[0] - 50, center[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calculate the raw AVERAGED position for THIS frame
        if len(global_estimates) > 0:
            raw_avg_x = sum(pos[0] for pos in global_estimates) / len(global_estimates)
            raw_avg_y = sum(pos[1] for pos in global_estimates) / len(global_estimates)
            
            # Add this frame's raw data to our rolling history
            history_x.append(raw_avg_x)
            history_y.append(raw_avg_y)
            
            # Calculate the SMOOTHED position across the last N frames
            smoothed_x = sum(history_x) / len(history_x)
            smoothed_y = sum(history_y) / len(history_y)

            # detect is moving
            smoothed_x_25 = sum(list(history_x)[-25:]) / len(list(history_x)[-25:])
            smoothed_y_25 = sum(list(history_y)[-25:]) / len(list(history_y)[-25:])
            if len(history_x) == 100:
                old_smoothed_x_25 = sum(list(history_x)[-50:-25]) / len(list(history_x)[-50:-25])
                old_smoothed_y_25 = sum(list(history_y)[-50:-25]) / len(list(history_y)[-50:-25])
            else:
                old_smoothed_x_25 = smoothed_x_25
                old_smoothed_y_25 = smoothed_x_25
            diff_x_25 = abs(smoothed_x_25 - old_smoothed_x_25)
            diff_y_25 = abs(smoothed_y_25 - old_smoothed_y_25)

            # Display the smooth coordinate
            if diff_x_25 < 0.015 or diff_y_25 < 0.015:
                # if physically still

                # detect avg 100 has settled **change to standard deviation instead? i guess for everything?
                first_x_25 = sum(list(history_x)[-25:]) / len(list(history_x)[-25:])
                first_y_25 = sum(list(history_y)[-25:]) / len(list(history_y)[-25:])
                if len(history_x) > 25:
                    last_x_25 = sum(list(history_x)[:25]) / len(list(history_x)[:25])
                    last_y_25 = sum(list(history_y)[:25]) / len(list(history_y)[:25])
                else:
                    last_x_25 = first_x_25
                    last_y_25 = first_x_25
                diff_x_100 = abs(first_x_25 - last_x_25)
                diff_y_100 = abs(first_y_25 - last_y_25)

                # if avg 100 has settled: dark green [use normal]
                if diff_x_100 < 0.014 or diff_y_100 < 0.014:
                    final_text = f"SUPER STILL POS: X={smoothed_x*100:.2f}cm, Y={smoothed_y*100:.2f}cm"
                    cv2.putText(frame, final_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
                    last_x = smoothed_x
                    last_y = smoothed_y
                else: # if avg 100 has not settled yet: green [use combo normal+fast]
                    final_text = f"STILL POS: X={smoothed_x*100:.2f}cm, Y={smoothed_y*100:.2f}cm"
                    cv2.putText(frame, final_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    last_x = smoothed_x*0.3+smoothed_x_25*0.7
                    last_y = smoothed_y*0.3+smoothed_y_25*0.7
            else:
                # if moving: yellow [use the faster changing (but more noisy) 10 one]
                final_text = f"MOVE POS: X={smoothed_x_25*100:.2f}cm, Y={smoothed_y_25*100:.2f}cm"
                cv2.putText(frame, final_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                last_x = smoothed_x_25
                last_y = smoothed_y_25

        else:
            # if no tags: red
            final_text = f"LAST POS: X={last_x*100:.2f}cm, Y={last_y*100:.2f}cm"
            cv2.putText(frame, final_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("AprilTag Localization", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()