import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 1,
        min_detection_confidence = 0.8,
        min_tracking_confidence = 0.5,
        model_complexity = 0
    )

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    while not cap.isOpened():
        cap.open(0)
        if time.time() - start_time > 10:
            return
        time.sleep(0.5)
    
    w_cam, h_cam = 640, 480
    cap.set(3, w_cam)
    cap.set(4, h_cam)

    # Screen dimensions
    screen_w, screen_h = pyautogui.size()

    frame_r = 100
    p_time = 0

    # Smoothing factors
    smooth_factor = 5
    ploc_x, ploc_y = 0, 0
    cloc_x, cloc_y = 0, 0


    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Draw active zone
        cv2.rectangle(frame, (frame_r, frame_r), (w_cam - frame_r, h_cam - frame_r), (255, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_lm, 
                    mp_hands.HAND_CONNECTIONS
                )

                # Index finger
                index = hand_lm.landmark[8]
                cx, cy = int(index.x * w_cam), int(index.y * h_cam)

                # Coordinate mapping
                x_mouse = np.interp(cx, (frame_r, w_cam - frame_r), (0, screen_w))
                y_mouse = np.interp(cy, (frame_r, h_cam - frame_r), (0, screen_h))

                # Smoothing
                cloc_x = ploc_x + (x_mouse - ploc_x) / smooth_factor
                cloc_y = ploc_y + (y_mouse - ploc_y) / smooth_factor

                # Move the actual cursor
                pyautogui.moveTo(cloc_x, cloc_y, _pause=False)

                ploc_x, ploc_y = cloc_x, cloc_y

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cv2.imshow('Milestone 2.1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

