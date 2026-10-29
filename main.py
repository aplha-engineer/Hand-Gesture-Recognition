import os
import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import pandas as pd

MODEL_PATH = os.path.join("model", "gesture_model.pkl")
CONFIDENCE_THRESHOLD = 0.70   # isse kam confidence par "Unknown" dikhega
SMOOTHING_FRAMES = 7          # itne frames ka majority vote

COLUMNS = [f"f{i}" for i in range(63)]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def draw_professional_ui(frame, gesture, confidence, fps):
    h, w = frame.shape[:2]

    # Header
    cv2.rectangle(frame, (0, 0), (w, 82), (25, 20, 14), -1)
    cv2.putText(
        frame, "HAND GESTURE RECOGNITION",
        (20, 34), cv2.FONT_HERSHEY_DUPLEX,
        0.75, (255, 190, 70), 2, cv2.LINE_AA
    )
    cv2.putText(
        frame, "OpenCV | MediaPipe | Machine Learning",
        (22, 61), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (200, 210, 220), 1, cv2.LINE_AA
    )

    # Prediction panel
    cv2.rectangle(frame, (14, 98), (min(w - 14, 360), 190),
                  (25, 38, 30), -1)
    cv2.rectangle(frame, (14, 98), (min(w - 14, 360), 190),
                  (80, 220, 145), 2)

    cv2.putText(frame, "CURRENT GESTURE", (28, 123),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (150, 230, 180), 1, cv2.LINE_AA)

    cv2.putText(frame, str(gesture).replace("_", " ")[:22], (28, 154),
                cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (100, 255, 180), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                (28, 178), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (240, 240, 240), 1, cv2.LINE_AA)

    # FPS indicator
    cv2.rectangle(frame, (w - 125, 98), (w - 14, 137),
                  (35, 30, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 112, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 210, 90), 2, cv2.LINE_AA)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 34), (w, h), (25, 20, 14), -1)
    cv2.putText(frame, "LIVE CAMERA  |  Press Q to quit",
                (16, h - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 225, 235), 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(frame, (2, 2), (w - 3, h - 3),
                  (220, 145, 45), 2)

    return frame



def extract_features(hand_landmarks):
    """collect_data.py wala EXACT same function (training se match hona zaroori hai)."""
    points = []

    for landmark in hand_landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])

    # Wrist (landmark 0) becomes the origin.
    wrist = points[0]
    normalized = []

    for point in points:
        normalized.extend([
            point[0] - wrist[0],
            point[1] - wrist[1],
            point[2] - wrist[2]
        ])

    # Scale normalization helps with different hand sizes.
    scale = max(
        (x * x + y * y + z * z) ** 0.5
        for x, y, z in [
            (normalized[i], normalized[i + 1], normalized[i + 2])
            for i in range(0, len(normalized), 3)
        ]
    )

    if scale > 0:
        normalized = [value / scale for value in normalized]

    return normalized


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model nahi mila. Pehle train_model.py chalao.")
        return

    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera open nahi hua. Camera index ya permission check karo.")
        return

    history = deque(maxlen=SMOOTHING_FRAMES)
    prev_time = time.time()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame read nahi hua.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            text = "No hand"
            color = (0, 0, 255)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                features = extract_features(hand)
                # DataFrame use kiya taaki sklearn feature-name warning na aaye
                X = pd.DataFrame([features], columns=COLUMNS)

                probs = model.predict_proba(X)[0]
                best = probs.argmax()
                label = model.classes_[best]
                confidence = probs[best]

                if confidence >= CONFIDENCE_THRESHOLD:
                    history.append(label)
                else:
                    history.append("Unknown")

                # Majority vote se flicker kam hota hai
                stable_label = Counter(history).most_common(1)[0][0]

                text = f"{stable_label} ({confidence * 100:.0f}%)"
                color = (0, 255, 0) if stable_label != "Unknown" else (0, 165, 255)
            else:
                history.clear()

            
            # FPS
            now = time.time()
            fps = 1 / max(now - prev_time, 1e-6)
            prev_time = now

            # Professional UI
            if results.multi_hand_landmarks:
                display_label = stable_label
                display_confidence = confidence * 100
            else:
                display_label = "No Hand"
                display_confidence = 0.0

            frame = draw_professional_ui(
                frame,
                display_label,
                display_confidence,
                fps
            )

            cv2.imshow("Hand Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()