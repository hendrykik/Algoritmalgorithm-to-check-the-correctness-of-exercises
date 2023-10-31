import cv2
import mediapipe as mp
from main import point_names
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def main():
    cap = cv2.VideoCapture('//Users/janzemlo/Inzynierka/squat_vids/front1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        tab_hip, tab_heel, tab_foot_index, tab_knee, tab_shoulder = [], [], [], [], []
        move_started = False
        first_squat = False
        max_depth_hip = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                h, w, c = frame.shape
                if not move_started:
                    if not check_move(tab_hip, tab_foot_index, tab_heel, fps, w, h):

                        tab_hip.append((
                            int(results.pose_landmarks.landmark[23].x * w),
                            int(results.pose_landmarks.landmark[23].y * h),
                            int(results.pose_landmarks.landmark[24].x * w),
                            int(results.pose_landmarks.landmark[24].y * h)
                        ))

                        tab_heel.append((
                            int(results.pose_landmarks.landmark[29].x * w),
                            int(results.pose_landmarks.landmark[29].y * h),
                            int(results.pose_landmarks.landmark[30].x * w),
                            int(results.pose_landmarks.landmark[30].y * h)
                        ))

                        tab_foot_index.append((
                            int(results.pose_landmarks.landmark[31].x * w),
                            int(results.pose_landmarks.landmark[31].y * h),
                            int(results.pose_landmarks.landmark[32].x * w),
                            int(results.pose_landmarks.landmark[32].y * h)
                        ))

                        tab_shoulder = [
                            int(results.pose_landmarks.landmark[11].x * w),
                            int(results.pose_landmarks.landmark[12].x * w)]
                    else:
                        move_started = True
                        check_feet(tab_heel[-1], tab_foot_index[-1], tab_shoulder)
                elif not check_max_depth(tab_hip, fps):
                    tab_hip.append((
                        int(results.pose_landmarks.landmark[23].x * w),
                        int(results.pose_landmarks.landmark[23].y * h),
                        int(results.pose_landmarks.landmark[24].x * w),
                        int(results.pose_landmarks.landmark[24].y * h)
                    ))
                    if max_depth_hip < (results.pose_landmarks.landmark[23].y * h):
                        max_depth_hip = results.pose_landmarks.landmark[23].y * h
                        tab_shoulder = [
                            int(results.pose_landmarks.landmark[11].x * w),
                            int(results.pose_landmarks.landmark[12].x * w)]
                        tab_knee = [
                            int(results.pose_landmarks.landmark[25].x * w),
                            int(results.pose_landmarks.landmark[26].x * w)]

                elif not first_squat:
                    print("koniec przysiadu")
                    first_squat = True
                    check_knee(tab_knee, tab_shoulder)

                print_skielet(results, frame, h, w)

            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def calculate_angle(x1, y1, x2, y2):
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return 90 - abs(angle_deg)


def check_max_depth(tab_hip, fps):
    before = int(len(tab_hip) - fps / 2)
    if before > 0:
        threshold_percentage_hip = 3
        if abs(tab_hip[-1][1] - tab_hip[before][1]) < abs(tab_hip[before][1]) * threshold_percentage_hip / 100:
            return True
    return False


def check_knee(tab_knee, tab_shoulder):
    # print(tab_knee, tab_shoulder)
    if (tab_knee[0] > tab_shoulder[0] and
            tab_knee[1] < tab_shoulder[1]
    ):
        print("Dobrze kolana są szerzej niż ramiona")


def check_feet(tab_heel, tabl_foot_index, tab_shoulder):
    angle_left = int(calculate_angle(tabl_foot_index[2], tabl_foot_index[3], tab_heel[2], tab_heel[3]))
    angle_right = int(calculate_angle(tab_heel[0], tab_heel[1], tabl_foot_index[0], tabl_foot_index[1]))

    print(f"Kąt lewej stopy wynosi {angle_left}")
    print(f"Kąt prawej stopy wynosi {angle_right}")

    threshold_percentage = 10
    if abs(abs(tab_heel[2] - tab_heel[0])) - abs(tab_shoulder[1] - tab_shoulder[0]) < abs(tab_shoulder[1] - tab_shoulder[0]) * threshold_percentage / 100:
        print("Dobrze ramiona i stopy są na takiej samej szerokosci")

    # print((abs(tab_heel[2] - tab_heel[0])), abs(tab_shoulder[1] - tab_shoulder[0]))
    # print(abs(tab_shoulder[1] - tab_shoulder[0]) * threshold_percentage / 100)
    #print(abs(tab_heel[0] - tab_shoulder[0]), abs(tab_shoulder[0]) * threshold_percentage / 100)


def check_move(tab_hip, tab_heel, tab_foot_index, fps, w, h):
    before = int(len(tab_hip) - fps / 2)
    if before > 0:
        threshold_percentage = 0.7
        if (abs(tab_heel[-1][0] - tab_heel[before][0]) < abs(tab_heel[before][0]) * threshold_percentage / 100 and
                abs(tab_heel[-1][1] - tab_heel[before][1]) < abs(tab_heel[before][1]) * threshold_percentage / 100 and
                abs(tab_heel[-1][2] - tab_heel[before][2]) < abs(tab_heel[before][2]) * threshold_percentage / 100 and
                abs(tab_heel[-1][3] - tab_heel[before][3]) < abs(tab_heel[before][3]) * threshold_percentage / 100
        ):
            threshold_percentage_hip = 3
            # print(f"diff =  {diff}, hip = {tab_hip[-1][1]}, hip_before = {tab_hip[before][1]}, heel = {tab_heel[-1][1]}")
            # print(tab_hip[-1][1], tab_hip[before][1])
            # print(tab_hip[-1][1] - tab_hip[before][1], diff / 6)
            if abs(tab_hip[-1][1] - tab_hip[before][1]) > abs(tab_hip[before][1]) * threshold_percentage_hip / 100:
                return True
    return False


def print_skielet(results, frame, h, w):
    for id, landmark in enumerate(results.pose_landmarks.landmark):
        if id in {11, 12, 25, 26, 23, 24, 29, 30, 31, 32}:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            point_name = point_names.get(id, "unknown")
            cv2.putText(frame, point_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
