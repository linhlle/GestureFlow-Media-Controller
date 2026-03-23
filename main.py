import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
import math

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

def main():

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )

    recognizer = GestureRecognizer.create_from_options(options)


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
    smooth_factor = 7
    ploc_x, ploc_y = 0, 0
    cloc_x, cloc_y = 0, 0
    last_toggle_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        # The Task API requires MediaPipe Image objects
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # In VIDEO mode, you must provide a monotonically increasing timestamp (ms)
        timestamp_ms = int(time.time() * 1000)
        # Perform asynchronous-like inference for the current frame
        results = recognizer.recognize_for_video(mp_image, timestamp_ms)

        cv2.rectangle(frame, (frame_r, frame_r), (w_cam - frame_r, h_cam - frame_r), (255, 0, 0), 2)

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            top_gesture = "None"
            if results.gestures and len(results.gestures) > 0:
                top_gesture = results.gestures[0][0].category_name

            index_tip = landmarks[8]
            ix, iy = int(index_tip.x * w_cam), int(index_tip.y * h_cam) 

            click_gestures = ["Victory", "Thumb_Up"]

            if top_gesture not in click_gestures:
                # Map from box to full screen
                x_target = np.interp(ix, (frame_r, w_cam - frame_r), (0, screen_w))
                y_target = np.interp(iy, (frame_r, h_cam - frame_r), (0, screen_h + 50))

                # Apply Exponential Smoothing (from Milestone 2.2)
                cloc_x = ploc_x + (x_target - ploc_x) / smooth_factor
                cloc_y = ploc_y + (y_target - ploc_y) / smooth_factor

                pyautogui.moveTo(cloc_x, cloc_y, _pause=False)
                ploc_x, ploc_y = cloc_x, cloc_y

                # B. GESTURE COMMANDS
            
            curr_time = time.time()

            if top_gesture == "Open_Palm" and (curr_time - last_toggle_time > 1.2):
                pyautogui.hotkey('command', 'option', 'd')
                last_toggle_time = curr_time
                cv2.putText(frame, "DOCK TOGGLE", (200, 400), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
            elif top_gesture in click_gestures:
                cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()
                time.sleep(0.1)
            cv2.putText(frame, f'Gesture: {top_gesture}', (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cv2.imshow('Milestone 2.1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

