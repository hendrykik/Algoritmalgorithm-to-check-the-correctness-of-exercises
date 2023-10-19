import cv2
import mediapipe as mp

# Inicjalizacja obiektu MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Słownik do mapowania indeksów na nazwy punktów
point_names = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

def main():
    # Inicjalizacja kamery wideo
    cap = cv2.VideoCapture('/Users/janzemlo/Desktop/Inzynierka/krotkie.mp4')

    # Inicjalizacja modelu śledzenia postawy
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Odczyt klatki z kamery
            ret, frame = cap.read()
            if not ret:
                break

            # Konwersja klatki na format RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detekcja postawy na podstawie klatki
            results = pose.process(frame_rgb)

            # Rysowanie szkieletu na obrazie wraz z nazwami punktów
            if results.pose_landmarks:
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    point_name = point_names.get(id, "unknown")
                    cv2.putText(frame, point_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Wyświetlenie klatki z szkieletem
            cv2.imshow('Pose Estimation', frame)

            # Przerwanie pętli po naciśnięciu klawisza 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Zwolnienie zasobów
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
