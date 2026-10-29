
import cv2
import mediapipe as mp
import pandas as pd
import os

DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "gestures.csv")

os.makedirs(DATA_DIR, exist_ok=True)

GESTURES = {
    1: "Open_Palm",
    2: "Fist",
    3: "Peace",
    4: "Thumbs_Up",
    5: "Pointing_Up"
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_features(hand_landmarks):
    points = []

    for landmark in hand_landmarks.landmark:
        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

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
        (x*x + y*y + z*z) ** 0.5
        for x, y, z in [
            (normalized[i], normalized[i+1],
             normalized[i+2])
            for i in range(0, len(normalized), 3)
        ]
    )

    if scale > 0:
        normalized = [value / scale for value in normalized]

    return normalized


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi hua. Camera index ya permission check karo.")
    raise SystemExit

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    selected_gesture = None
    saved_count = 0

    if os.path.exists(CSV_FILE):
        saved_count = len(pd.read_csv(CSV_FILE))

    print("1-5: Gesture select | S: Save sample | Q: Quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame read nahi hua.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )
            features = extract_features(hand)

        cv2.putText(
            frame,
            "1:Open 2:Fist 3:Peace 4:Thumbs 5:Point",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

        label_text = selected_gesture or "Not selected"
        cv2.putText(
            frame, f"Selected: {label_text}",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"Saved samples: {saved_count}",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

        cv2.imshow("Collect Gesture Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(i)) for i in range(1, 6)]:
            selected_gesture = GESTURES[int(chr(key))]
            print("Selected:", selected_gesture)

        elif key == ord("s"):
            if selected_gesture is None:
                print("Pehle 1-5 se gesture select karo.")
            elif features is None:
                print("Hand detect nahi hua. Dobara try karo.")
            else:
                row = features + [selected_gesture]
                columns = [
                    f"f{i}" for i in range(63)
                ] + ["label"]

                pd.DataFrame(
                    [row], columns=columns
                ).to_csv(
                    CSV_FILE, mode="a",
                    header=not os.path.exists(CSV_FILE),
                    index=False
                )

                saved_count += 1
                print(
                    f"Saved: {selected_gesture} | "
                    f"Total samples: {saved_count}"
                )

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")
