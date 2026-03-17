import cv2
import mediapipe as mp

def main():

    # mp_hands provides the model; mp_draw provides the utility to draw the lines
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Configure the Hands object:
    # static_image_mode=False: Optimizes for video tracking (faster)
    # max_num_hands=1: Tracks only one hand to save CPU
    # min_detection_confidence: AI must be 70% sure it's a hand to start tracking
    # min_tracking_confidence: AI must be 50% sure it's the SAME hand to keep tracking
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5
    )
    
    # 0: std wecame. Can pass a video file path for pre-recorded footage
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam starts. Press 'q' to quit")

    while True:
        # Capture frame by frame
        # 'ret' is a boolean (True if frame captured), 'frame' is the image array
        ret, frame = cap.read()

        if not ret:
            print("Error to grab frame")
            break

        # Mirror the frame
        # 1: Horizontal; 0: Vertical; -1: Both
        frame = cv2.flip(frame, 1)

        # OpenCV uses BGR (Blue-Green-Red), but MediaPipe needs RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # perform detection
        results = hands.process(img_rgb)

        # Extract and Draw Landmarks
        # results.multi_hand_landmarks will contain the 21 points if a hand is found
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # This helper function draws the dots and lines automatically
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmark, 
                    mp_hands.HAND_CONNECTIONS
                )

                # To see the specific (x, y) of your Index Finger Tip (Point 8):
                index_tip = hand_landmark.landmark[8]
                # print(f"Index Tip: x={index_tip.x:.2f}, y={index_tip.y:.2f}")

        # cv2.imshow(window_name, image)
        cv2.imshow('GMC Phase 1.1', frame)

        # cv2.waitKey(1) waits 1ms for a key event. 0xFF mask captures the key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release() # close webcam hardware
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

