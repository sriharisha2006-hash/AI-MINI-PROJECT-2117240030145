import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    prev_x, prev_y = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0

        combo = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Air Writing", combo)

        key = cv2.waitKey(1)
        if key == ord('c'):
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
