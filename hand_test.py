import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Camera not detected")
        break

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, c = img.shape

    if results.multi_hand_landmarks:

        # Draw landmarks on both hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # If two hands are detected, connect matching landmarks
        if len(results.multi_hand_landmarks) == 2:

            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]

            for i in range(21):

                x1 = int(hand1.landmark[i].x * w)
                y1 = int(hand1.landmark[i].y * h)

                x2 = int(hand2.landmark[i].x * w)
                y2 = int(hand2.landmark[i].y * h)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                cv2.circle(img, (x1, y1), 5, (0, 255, 0), -1)
                cv2.circle(img, (x2, y2), 5, (0, 255, 0), -1)

    cv2.imshow("Two Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()