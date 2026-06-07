import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    if not success:
        print("Failed to read camera")
        break

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(img, detection)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()