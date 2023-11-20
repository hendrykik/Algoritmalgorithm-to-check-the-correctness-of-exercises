import cv2
import mediapipe as mp
from Implementation_of_a_pose_estimation_algorithm_and_application_for_the_analysis_of_exercise_correctness.point_names import point_names

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def main():
    cap = cv2.VideoCapture('//Users/janzemlo/Inzynierka/squat_vids/back1.mp4')

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                for id, landmark in enumerate(results.pose_landmarks.landmark):
                    if id in {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30}:
                        h, w, c = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        point_name = point_names.get(id, "unknown")
                        cv2.putText(frame, point_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                    cv2.LINE_AA)

            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def check_grip():


    print("Dobre ułożenie nagrastków")


if __name__ == "__main__":
    main()
