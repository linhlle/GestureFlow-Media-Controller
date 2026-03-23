import mediapipe as mp
import cv2
import time
import csv
import os

def main():
    mp_hands= mp.solutions.hands
    mp_draws = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_tracking_confidence=0.5,
        min_detection_confidence=0.8
    )

    cap = cv2.VideoCapture(0)
    while not cap.isOpened():
        cap.open(0)
        time.sleep(0.5)

    counts = {'0': 0, '1': 0, '2': 0, '3': 0}

    csv_path = 'gesture_data.csv'
    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            try:
                next(reader)
                for row in reader:
                    if row:
                        label = row[-1]
                        if label in counts:
                            counts[label] += 1
            except StopIteration:
                pass

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ =  frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)

        # On screen counter
        cv2.rectangle(frame, (0, 0), (220, 160), (0, 0, 0), -1)
        y_offset = 30
        for label, count in counts.items():
            color = (255, 255, 255) if count < 250 else (0, 255, 0)
            name = ["Neutral", "L-Shape", "High-Five", "2-Finger"][int(label)]
            cv2.putText(frame, f"{label} {name}: {count}", (10, y_offset), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
            y_offset += 30
    
        key = cv2.waitKey(1) & 0xFF

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draws.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                if ord('0') <= key <= ord('3'):
                    label = chr(key)
                    data_row = []
                    for lm in hand_lms.landmark:
                        data_row.extend([lm.x, lm.y, lm.z])

                    data_row.append(label)

                    file_exists = os.path.isfile(csv_path)
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            header = [f'lm{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']] + ['label']
                            writer.writerow(header)
                        writer.writerow(data_row)

                    counts[label] += 1

        cv2.imshow('Data Logger', frame)
        if key == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

