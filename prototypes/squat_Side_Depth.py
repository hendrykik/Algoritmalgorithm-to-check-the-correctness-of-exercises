import cv2
import mediapipe as mp
from point_names import point_names
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def squat_Side():
    cap = cv2.VideoCapture('//Users/janzemlo/Inzynierka/squat_vids/side1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_time = time.time()

    tab_hip, tab_heel, tab_foot_index, tab_knee, tab_shoulder_started, tab_shoulder, tab_shoulder_end = [], [], [], [], [], [], []
    move_started = False
    end_squat = False
    max_depth_hip = 0

    max_depth_hip = 0
    max_depth_knee = 0
    max_depth_knee_x = 0
    max_depth_foot_x = 0
    w = 0
    left = True

    tab_neck_sec = set()
    tab_feet_sec = set()

    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.95) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                h, w, c = frame.shape
                if check_side_left(results):
                    left = True
                else:
                    left = False

                print_skielet(results, frame, h, w, left)
                heel_index = 29 if left else 30

                if not move_started:
                    if not check_feet_before_start(tab_heel, tab_foot_index, fps):
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
                    else:
                        if not check_squat_started(tab_shoulder, tab_hip, fps):
                            tab_shoulder.append((
                                int(results.pose_landmarks.landmark[11].x * w),
                                int(results.pose_landmarks.landmark[11].x * h),
                                int(results.pose_landmarks.landmark[12].x * w),
                                int(results.pose_landmarks.landmark[12].x * h)
                            ))
                            tab_hip.append((
                                int(results.pose_landmarks.landmark[23].x * w),
                                int(results.pose_landmarks.landmark[23].y * h),
                                int(results.pose_landmarks.landmark[24].x * w),
                                int(results.pose_landmarks.landmark[24].y * h)
                            ))
                        else:
                            move_started = True
                            elapsed_time = time.time() - start_time
                            print(f"Start przysiadu {elapsed_time}")
                            tab_shoulder_started = [
                                int(results.pose_landmarks.landmark[11].x * w),
                                int(results.pose_landmarks.landmark[11].x * h),
                                int(results.pose_landmarks.landmark[12].x * w),
                                int(results.pose_landmarks.landmark[12].x * h)]
                            heel_before = [
                                int(results.pose_landmarks.landmark[heel_index].x * w),
                                int(results.pose_landmarks.landmark[heel_index].x * h)]

                elif not end_squat:
                    if not check_end_squat(tab_shoulder_end, tab_shoulder_started):
                        tab_shoulder_end = [
                            int(results.pose_landmarks.landmark[11].x * w),
                            int(results.pose_landmarks.landmark[11].x * h),
                            int(results.pose_landmarks.landmark[12].x * w),
                            int(results.pose_landmarks.landmark[12].x * h)]

                        nose = [
                            int(results.pose_landmarks.landmark[0].x * w),
                            int(results.pose_landmarks.landmark[0].x * h)]

                        shoulder_index = 11 if left else 12

                        shoulder = [
                            int(results.pose_landmarks.landmark[shoulder_index].x * w),
                            int(results.pose_landmarks.landmark[shoulder_index].y * h)
                        ]

                        hip_index = 23 if left else 24

                        hip = [
                            int(results.pose_landmarks.landmark[hip_index].x * w),
                            int(results.pose_landmarks.landmark[hip_index].y * h)
                        ]

                        if not check_neck(shoulder, hip, nose):
                            elapsed_time = time.time() - start_time
                            tab_neck_sec.add(int(elapsed_time))

                        heel = [
                            int(results.pose_landmarks.landmark[heel_index].x * w),
                            int(results.pose_landmarks.landmark[heel_index].x * h)]

                        if not check_tearing_off_feet(heel, heel_before):
                            elapsed_time = time.time() - start_time
                            tab_feet_sec.add(int(elapsed_time))

                    else:
                        elapsed_time = time.time() - start_time
                        print(f"Koniec przysiadu {elapsed_time}")
                        end_squat = True

            cv2.imshow('Pose Estimation', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        check_depth(max_depth_hip, max_depth_knee)
        check_shins(max_depth_knee_x, max_depth_foot_x, w)

        if len(tab_neck_sec) > 0:
            print("Złe ułożenie kręgosłupa w sekundach:", tab_neck_sec)
        else:
            print("Brak złego ułożenia kręgosłupa.")

        if len(tab_feet_sec) > 0:
            print("Oderwanie stóp w sekundach:", tab_feet_sec)
        else:
            print("Przysiad bez oderwania stóp.")

        cap.release()
        cv2.destroyAllWindows()


def check_squat_started(tab_shoulder, tab_hip, fps):
    before = int(len(tab_hip) - fps)
    if before > 0:
        threshold_percentage_hip = 3
        threshold_percentage_shoulder = 3

        diff_hip = tab_hip[-1][1] - tab_hip[before][1]
        diff_shoulder = tab_shoulder[-1][1] - tab_shoulder[before][1]

        if (
                abs(diff_hip) > abs(tab_hip[before][1]) * threshold_percentage_hip / 100 and
                abs(diff_shoulder) > abs(tab_shoulder[before][1]) * threshold_percentage_shoulder / 100
        ):
            return True
    return False


def check_depth(max_depth_hip, max_depth_knee):
    if max_depth_hip < max_depth_knee:
        print("Poprawna głębokość przysiadu")
    else:
        print("Niepoprawna głębokość przysiadu")


def check_shins(max_depth_knee_x, max_depth_foot_x, w):
    threshold_percentage = w * 1.5 / 100
    if abs(max_depth_knee_x - max_depth_foot_x) < threshold_percentage:
        print("Poprawne ułożenie piszczela")
    else:
        print("Niepoprawne ułożenie piszczela")


def check_end_squat(tab_shoulder_end, tab_shoulder_started):
    if len(tab_shoulder_end) > 0:
        if (
                (tab_shoulder_end[1] >= tab_shoulder_started[1]) and
                (tab_shoulder_end[3] >= tab_shoulder_started[3])
        ):
            return True
    return False


def check_feet_before_start(tab_heel, tab_foot_index, fps):
    before = int(len(tab_heel) - fps / 2)  # Zwiększamy liczbę klatek przed aktualnym momentem
    if before > 0:
        threshold_percentage = 2
        if (
                abs(tab_heel[-1][0] - tab_heel[before][0]) < abs(tab_heel[before][0]) * threshold_percentage / 100 and
                abs(tab_heel[-1][1] - tab_heel[before][1]) < abs(tab_heel[before][1]) * threshold_percentage / 100 and
                abs(tab_heel[-1][2] - tab_heel[before][2]) < abs(tab_heel[before][2]) * threshold_percentage / 100 and
                abs(tab_heel[-1][3] - tab_heel[before][3]) < abs(tab_heel[before][3]) * threshold_percentage / 100 and
                abs(tab_foot_index[-1][0] - tab_foot_index[before][0]) < abs(
            tab_foot_index[before][0]) * threshold_percentage / 100 and
                abs(tab_foot_index[-1][1] - tab_foot_index[before][1]) < abs(
            tab_foot_index[before][1]) * threshold_percentage / 100 and
                abs(tab_foot_index[-1][2] - tab_foot_index[before][2]) < abs(
            tab_foot_index[before][2]) * threshold_percentage / 100 and
                abs(tab_foot_index[-1][3] - tab_foot_index[before][3]) < abs(
            tab_foot_index[before][3]) * threshold_percentage / 100
        ):
            # print("start przysiadu")
            return True
    return False


def check_side_left(results):
    if results.pose_landmarks.landmark[29].y > results.pose_landmarks.landmark[30].y:
        return True
    return False


def check_neck(shoulder, hip, nose):
    m = (hip[1] - shoulder[1]) / (hip[0] - shoulder[0])
    b = shoulder[1] - m * shoulder[0]

    y_nosea_na_prostej = m * nose[0] + b

    if min(shoulder[1], hip[1]) <= y_nosea_na_prostej <= max(shoulder[1], hip[1]):
        return True
    else:
        return False


def check_max_depth(tab_hip, fps):
    before = int(len(tab_hip) - fps)
    if before > 0:
        if tab_hip[-1][1] < tab_hip[before][1]:
            return True
    return False


def check_tearing_off_feet(heel, heel_before):
    threshold_percent = 1
    y_diff = heel[1] - heel_before[1]

    threshold = abs(heel_before[1] * threshold_percent / 100)

    if y_diff < threshold:
        return True
    else:
        return False


def print_skielet(results, frame, h, w, left):
    if left:
        ids = {11, 25, 23, 29, 31}
    else:
        ids = {12, 26, 24, 30, 32}
    for id, landmark in enumerate(results.pose_landmarks.landmark):
        if id in ids:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            point_name = point_names.get(id, "unknown")
            cv2.putText(frame, point_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    squat_Side()