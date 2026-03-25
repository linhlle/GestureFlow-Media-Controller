import mediapipe as mp
import cv2
import time
import pyautogui
import pickle
import numpy as np
import os
import subprocess
from collections import deque, Counter
from utils import normalize_landmarks, get_current_volume

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SMOOTH_FACTOR = 5

VOTE_WINDOW_SIZE = 10
VOTE_THRESHOLD = 7
CONFIDENCE_THRESHOLD = 0.8

GESTURE_MAP = {
    1: "Spotlight",     # L-Shape
    2: "Mission Ctrl",  # High-Five
    3: "App Switcher"   # 2-Finger Point
}


def main():
    try:
        with open('gesture_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Error: Have not trained the model")
        return
    
    current_vol = get_current_volume()
    last_vol_update_time = 0

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.open(0)
        time.sleep(0.5)

    w_cam, h_cam = 640, 480
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    screen_w, screen_h = pyautogui.size()
    frame_r = 100
    ploc_x, ploc_y = 0, 0
    last_cmd_time = 0
    prev_y = 0

    vote_history = deque(maxlen=VOTE_WINDOW_SIZE)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        cv2.rectangle(frame, (frame_r, frame_r), (w_cam-frame_r, h_cam-frame_r), (255, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                landmarks = hand_lm.landmark
                processed_data = normalize_landmarks(landmarks)

                probs = model.predict_proba([processed_data])[0]
                pred = np.argmax(probs)
                confidence = probs[pred]
                
                label = int(pred)

                if confidence > CONFIDENCE_THRESHOLD:
                    vote_history.append(pred)
                else: 
                    vote_history.append(0)

                vote_counts = Counter(vote_history).most_common(1)
                stable_pred = vote_counts[0][0]
                vote_score = vote_counts[0][1]

                index_tip = hand_lm.landmark[8]
                ix, iy = int(index_tip.x * w_cam), int(index_tip.y * h_cam)

                current_action = "Hovering"
                if vote_score >= VOTE_THRESHOLD and stable_pred != 0:
                    curr_time = time.time()
                    if curr_time - last_cmd_time > 1.3:
                        if stable_pred == 1: #Spotloght
                            pyautogui.hotkey('command', 'space')
                        elif stable_pred == 2: # Mission control
                            pyautogui.hotkey('ctrl', 'up')
                        elif stable_pred == 3: # App switcher
                            pyautogui.hotkey('command', 'tab')
                        
                        last_cmd_time = curr_time
                        vote_history.clear()
                    current_action = f"GESTURE {stable_pred} DETECTED"

                else:
                    x_target = np.interp(ix, (frame_r, w_cam - frame_r), (0, screen_w))
                    y_target = np.interp(iy, (frame_r, h_cam - frame_r), (0, screen_h + 50))
                    
                    cloc_x = ploc_x + (x_target - ploc_x) / SMOOTH_FACTOR
                    cloc_y = ploc_y + (y_target - ploc_y) / SMOOTH_FACTOR
                    pyautogui.moveTo(cloc_x, cloc_y, _pause=False)
                    ploc_x, ploc_y = cloc_x, cloc_y


                # Thumb: slider logic
                thumb_tip = hand_lm.landmark[4]
                if thumb_tip.y < index_tip.y and stable_pred == 0:
                    curr_y = hand_lm.landmark[0].y
                    if prev_y != 0:
                        diff = prev_y - curr_y
                        if abs(diff) > 0.02:
                            change = 5 if diff > 0 else -5
                            subprocess.run(["osascript", "-e", f"set volume output volume (output volume of (get volume settings) + {change})"])
                            current_vol = get_current_volume()

                    prev_y = curr_y
                    bar_y = int(np.interp(current_vol, [0, 100], [400, 150]))
                    cv2.rectangle(frame, (590, 150), (610, 400), (50, 50, 50), -1) # Background
                    cv2.rectangle(frame, (590, bar_y), (610, 400), (0, 255, 255), -1) # Progress
                    cv2.putText(frame, f"{current_vol}%", (580, 430), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
                else:
                    prev_y = 0


                # msg = GESTURE_MAP.get(label, "Moving...")
                color = (0, 255, 0) if stable_pred != 0 else (255, 255, 255)
                cv2.putText(frame, f"{current_action} (Consensus: {vote_score}/{VOTE_WINDOW_SIZE})", 
                            (20, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)                
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('AeroPoint Engine', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()