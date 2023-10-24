import cv2
import mediapipe as mp
from main import point_names

# Inicjalizacja obiektu MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    # Inicjalizacja kamery wideo
    cap = cv2.VideoCapture('//Users/janzemlo/Inzynierka/krotkie.mp4')
    max_depth_hip = 0
    max_depth_knee = 0

    # Inicjalizacja modelu śledzenia postawy
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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
                    if id in {25, 27, 29, 11, 23}:
                        h, w, c = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        point_name = point_names.get(id, "unknown")
                        cv2.putText(frame, point_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        if id == 23:
                            if max_depth_hip < int(cy):
                                max_depth_hip = int(cy)
                                max_depth_knee = int(results.pose_landmarks.landmark[25].y * h)
                                # print(max_depth_hip)
                                # print(int(results.pose_landmarks.landmark[25].y * h))
                                #max_depth_knee = int(cy)

            # Wyświetlenie klatki z szkieletem
            cv2.imshow('Pose Estimation', frame)

            # Przerwanie pętli po naciśnięciu klawisza 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Zwolnienie zasobów
        print(max_depth_hip)
        print(max_depth_knee)

        if max_depth_hip < max_depth_knee:
            print("Poprawna głębokość przysiadu")
        else:
            print("Niepoprawna głębokość przysiadu")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
