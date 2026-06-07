import cv2
import mediapipe as mp

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# High Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Full Screen Window
cv2.namedWindow("Face Mesh + Connected Hands", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Face Mesh + Connected Hands",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Mesh
    face_results = face_mesh.process(rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    # Hand Detection
    hand_results = hands.process(rgb)

    h, w, _ = frame.shape

    if hand_results.multi_hand_landmarks:

        # Draw both hands
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # Connect landmarks between two hands
        if len(hand_results.multi_hand_landmarks) == 2:

            hand1 = hand_results.multi_hand_landmarks[0]
            hand2 = hand_results.multi_hand_landmarks[1]

            for i in range(21):

                x1 = int(hand1.landmark[i].x * w)
                y1 = int(hand1.landmark[i].y * h)

                x2 = int(hand2.landmark[i].x * w)
                y2 = int(hand2.landmark[i].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)

    cv2.imshow("Face Mesh + Connected Hands", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()