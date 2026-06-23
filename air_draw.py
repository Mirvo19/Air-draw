import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import time
import math

MP_TASK_FILE = 'hand_landmarker.task'

def download_model():
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    print(f"Downloading {MP_TASK_FILE} from {url}...")
    try:
        urllib.request.urlretrieve(url, MP_TASK_FILE) 
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        exit()

if not os.path.exists(MP_TASK_FILE):
    download_model()

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_TASK_FILE),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5, 
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)    

landmarker = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)

img_canvas = None

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0
draw_color = colors[0]

hand_states = {}
SMOOTHING_FACTOR = 3 
LANDMARK_SMOOTHING_ALPHA = 0.8 
HAND_HOLD_FRAMES = 4 
FINGER_HYSTERESIS = 0.05 

print("Air Canvas started (using Tasks API)")
print("Controls:")
print("- Index finger up: Move Brush")
print("- Pinch (Index + Thumb): Lift Brush (Pause Drawing)")
print("- Index + Middle fingers up: Select Color / Clear")
print("- Press 'q' to quit")

start_time = time.time()
p_time = 0

def is_finger_extended(lm_list, tip_idx, pip_idx, wrist_idx=0, hysteresis=0.0):
    """
    Checks if a finger is extended by comparing distance from wrist to tip 
    vs wrist to PIP joint. Rotation invariant.
    """
    wx, wy = lm_list[wrist_idx][1], lm_list[wrist_idx][2]
    tx, ty = lm_list[tip_idx][1], lm_list[tip_idx][2]
    px, py = lm_list[pip_idx][1], lm_list[pip_idx][2]
    
    dist_tip = math.hypot(tx - wx, ty - wy)
    dist_pip = math.hypot(px - wx, py - wy)
    
    return dist_tip > (dist_pip * (1 + hysteresis))

def smooth_landmarks(raw_lm_list, prev_lm_list, alpha):
    if prev_lm_list is None or len(prev_lm_list) != len(raw_lm_list):
        return raw_lm_list, raw_lm_list

    smoothed = []
    for idx, (_, x, y) in enumerate(raw_lm_list):
        prev_x, prev_y = prev_lm_list[idx][1], prev_lm_list[idx][2]
        sx = int(prev_x + (x - prev_x) * alpha)
        sy = int(prev_y + (y - prev_y) * alpha)
        smoothed.append([idx, sx, sy])

    return smoothed, smoothed

def get_hand_key(result, hand_index):
    try:
        if result.handedness and len(result.handedness) > hand_index:
            label = result.handedness[hand_index][0].category_name
            return label if label else str(hand_index)
    except Exception:
        pass
    return str(hand_index)

def hsv_to_bgr(h, s, v):
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_cool_hand_skeleton(img, lm_list, hand_index):
    base_hue = 30 + (hand_index * 90) % 180
    glow_color = hsv_to_bgr(base_hue, 255, 255)
    core_color = hsv_to_bgr((base_hue + 20) % 180, 200, 255)

    for p1, p2 in HAND_CONNECTIONS:
        x1, y1 = lm_list[p1][1], lm_list[p1][2]
        x2, y2 = lm_list[p2][1], lm_list[p2][2]
        cv2.line(img, (x1, y1), (x2, y2), glow_color, 6)
        cv2.line(img, (x1, y1), (x2, y2), core_color, 2)

    for idx, (_, x, y) in enumerate(lm_list):
        radius = 3 + (idx % 4)
        joint_color = hsv_to_bgr((base_hue + idx * 6) % 180, 180, 255)
        cv2.circle(img, (x, y), radius + 2, glow_color, cv2.FILLED)
        cv2.circle(img, (x, y), radius, joint_color, cv2.FILLED)

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        img = cv2.flip(img, 1)
        
        if img_canvas is None:
            img_canvas = np.zeros_like(img)

        h, w, c = img.shape
        cv2.rectangle(img, (0, 0), (100, 65), (255, 0, 0), cv2.FILLED) # Blue
        cv2.rectangle(img, (100, 0), (200, 65), (0, 255, 0), cv2.FILLED) # Green
        cv2.rectangle(img, (200, 0), (300, 65), (0, 0, 255), cv2.FILLED) # Red
        cv2.rectangle(img, (300, 0), (400, 65), (0, 255, 255), cv2.FILLED) # Yellow
        cv2.rectangle(img, (400, 0), (500, 65), (255, 255, 255), cv2.FILLED) # Clear
        cv2.putText(img, "CLEAR", (410, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        if result.hand_landmarks:
            seen_keys = set()
            for hand_index, hand_lms in enumerate(result.hand_landmarks):
                hand_key = get_hand_key(result, hand_index)
                seen_keys.add(hand_key)

                if hand_key not in hand_states:
                    hand_states[hand_key] = {
                        "xp": 0,
                        "yp": 0,
                        "prev_x": 0,
                        "prev_y": 0,
                        "prev_landmarks": None,
                        "hold": 0
                    }

                state = hand_states[hand_key]

                raw_lm_list = []
                for id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    raw_lm_list.append([id, cx, cy])

                lm_list, state["prev_landmarks"] = smooth_landmarks(
                    raw_lm_list, state["prev_landmarks"], LANDMARK_SMOOTHING_ALPHA
                )
                state["hold"] = HAND_HOLD_FRAMES

                draw_cool_hand_skeleton(img, lm_list, hand_index)

                if len(lm_list) != 0:
                    raw_x, raw_y = lm_list[8][1:] 

                    if state["prev_x"] == 0 and state["prev_y"] == 0:
                        state["prev_x"], state["prev_y"] = raw_x, raw_y

                    curr_x = state["prev_x"] + (raw_x - state["prev_x"]) / SMOOTHING_FACTOR
                    curr_y = state["prev_y"] + (raw_y - state["prev_y"]) / SMOOTHING_FACTOR

                    x1, y1 = int(curr_x), int(curr_y) 
                    state["prev_x"], state["prev_y"] = curr_x, curr_y

                    cv2.circle(img, (x1, y1), 10, draw_color, cv2.FILLED)

                    index_up = is_finger_extended(lm_list, 8, 6, hysteresis=FINGER_HYSTERESIS)
                    middle_up = is_finger_extended(lm_list, 12, 10, hysteresis=FINGER_HYSTERESIS)

                    x_thumb, y_thumb = lm_list[4][1:]
                    pinch_dist = math.hypot(raw_x - x_thumb, raw_y - y_thumb)
                    palm_size = math.hypot(lm_list[0][1] - lm_list[9][1], lm_list[0][2] - lm_list[9][2])
                    pinch_threshold = max(20, palm_size * 0.35)

                    if index_up and middle_up:
                        state["xp"], state["yp"] = 0, 0

                        x_mid, y_mid = lm_list[12][1:]
                        cv2.rectangle(img, (x1, y1 - 25), (x_mid, y_mid + 25), draw_color, cv2.FILLED)

                        if y1 < 65:
                            if 0 < x1 < 100:
                                draw_color = (255, 60, 0)
                            elif 100 < x1 < 200:
                                draw_color = (0, 255, 0)
                            elif 200 < x1 < 300:
                                draw_color = (0, 0, 255)
                            elif 300 < x1 < 400:
                                draw_color = (0, 255, 255)
                            elif 400 < x1 < 500:
                                draw_color = (0, 0, 0)
                                img_canvas = np.zeros_like(img)

                    elif index_up and not middle_up:
                        if pinch_dist < pinch_threshold:
                            cv2.circle(img, (x1, y1), 15, (0, 0, 0), 2) 
                            cv2.putText(img, "Lifted", (x1+20, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                            state["xp"], state["yp"] = 0, 0 
                        else:
                            if state["xp"] == 0 and state["yp"] == 0:
                                state["xp"], state["yp"] = x1, y1

                            cv2.line(img_canvas, (state["xp"], state["yp"]), (x1, y1), draw_color, 15)
                            state["xp"], state["yp"] = x1, y1
                    else:
                        state["xp"], state["yp"] = 0, 0 

            missing_keys = [k for k in hand_states.keys() if k not in seen_keys]
            for k in missing_keys:
                if hand_states[k]["hold"] > 0 and hand_states[k]["prev_landmarks"] is not None:
                    hand_states[k]["hold"] -= 1
                    lm_list = hand_states[k]["prev_landmarks"]
                    draw_cool_hand_skeleton(img, lm_list, 0)
                else:
                    del hand_states[k]

        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, img_canvas)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (w - 190, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Air Canvas", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Loop Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()

