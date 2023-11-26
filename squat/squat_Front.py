import cv2
import mediapipe as mp
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def squat_Front(video_path, callback):
    #video_path = '//Users/janzemlo/Inzynierka/squat_vids/front1.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()

    tabs = initialize_tabs()
    squat_count, max_depth_hip = 0, 0
    squat_started, squat_ended, squat_completed = False, False, False
    tab_knee, tab_shoulder = [], []

    print_tips()

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = read_frame(cap)
            if not ret:
                break

            results = process_frame(frame, pose)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                squat_started, squat_ended, squat_completed = process_squat_phases(results, tabs, fps, w, h, start_time,
                                                                                   squat_started, squat_ended,
                                                                                   squat_count)

                if squat_started and not squat_ended:
                    max_depth_hip, tab_knee, tab_shoulder = update_max_depth(results, max_depth_hip, tab_knee,
                                                                             tab_shoulder, w, h)

                if squat_completed:
                    squat_count += 1
                    print(f"Początek {squat_count + 1} przysiadu w {tabs['squat_start_time']} sekundzie.")
                    print(f"Koniec {squat_count + 1} przysiadu w {tabs['squat_end_time']} sekundzie.")
                    check_knee(tab_knee, tab_shoulder)
                    tabs = initialize_tabs()
                    squat_started, squat_ended = False, False

                print_skeleton(results, frame)
            callback(frame)

            #display_frame(frame)

            if exit_requested():
                break
        
        print_summary(squat_count, start_time)
        cleanup(cap)


def process_squat_phases(results, tabs, fps, w, h, start_time, squat_started, squat_ended, squat_count):
    squat_completed = False

    if not squat_started:
        if not check_feet_before_start(tabs['heel'], tabs['foot_index'], fps):
            tabs['heel'].append((
                int(results.pose_landmarks.landmark[29].x * w),
                int(results.pose_landmarks.landmark[29].y * h),
                int(results.pose_landmarks.landmark[30].x * w),
                int(results.pose_landmarks.landmark[30].y * h)
            ))

            tabs['foot_index'].append((
                int(results.pose_landmarks.landmark[31].x * w),
                int(results.pose_landmarks.landmark[31].y * h),
                int(results.pose_landmarks.landmark[32].x * w),
                int(results.pose_landmarks.landmark[32].y * h)
            ))
        else:
            if not check_squat_started(tabs['shoulder'], tabs['hip'], fps):
                tabs['shoulder'].append((
                    int(results.pose_landmarks.landmark[11].x * w),
                    int(results.pose_landmarks.landmark[11].y * h),
                    int(results.pose_landmarks.landmark[12].x * w),
                    int(results.pose_landmarks.landmark[12].y * h)
                ))
                tabs['hip'].append((
                    int(results.pose_landmarks.landmark[23].x * w),
                    int(results.pose_landmarks.landmark[23].y * h),
                    int(results.pose_landmarks.landmark[24].x * w),
                    int(results.pose_landmarks.landmark[24].y * h)
                ))
            else:
                squat_started = True
                tabs['squat_start_time'] = time.time() - start_time  # Record the start time of the squat
                check_feet(tabs['heel'][-1], tabs['foot_index'][-1], tabs['shoulder'])
                tabs['shoulder_started'] = [
                    int(results.pose_landmarks.landmark[11].x * w),
                    int(results.pose_landmarks.landmark[11].y * h),
                    int(results.pose_landmarks.landmark[12].x * w),
                    int(results.pose_landmarks.landmark[12].y * h)
                ]
                tabs['hip_started'] = [
                    int(results.pose_landmarks.landmark[11].x * w),
                    int(results.pose_landmarks.landmark[11].y * h),
                    int(results.pose_landmarks.landmark[12].x * w),
                    int(results.pose_landmarks.landmark[12].y * h)
                ]
                tabs['heel_before'] = [
                    int(results.pose_landmarks.landmark[29].x * w),
                    int(results.pose_landmarks.landmark[29].y * h),
                    int(results.pose_landmarks.landmark[30].x * w),
                    int(results.pose_landmarks.landmark[30].y * h)
                ]

    elif not squat_ended:
        if not check_squat_ended(tabs['shoulder_end'], tabs['shoulder_started'],
                                 tabs['hip_end'], tabs['hip_started']):
            tabs['shoulder_end'] = [
                int(results.pose_landmarks.landmark[11].x * w),
                int(results.pose_landmarks.landmark[11].y * h),
                int(results.pose_landmarks.landmark[12].x * w),
                int(results.pose_landmarks.landmark[12].y * h)
            ]

            tabs['hip_end'] = [
                int(results.pose_landmarks.landmark[11].x * w),
                int(results.pose_landmarks.landmark[11].y * h),
                int(results.pose_landmarks.landmark[12].x * w),
                int(results.pose_landmarks.landmark[12].y * h)
            ]
        else:
            squat_ended = True
            squat_completed = True
            tabs['squat_end_time'] = time.time() - start_time

    return squat_started, squat_ended, squat_completed


def initialize_tabs():
    return {
        'hip': [], 'heel': [], 'foot_index': [], 'knee': [],
        'shoulder': [], 'shoulder_started': [], 'shoulder_end': [],
        'hip_started': [], 'hip_end': [],
        'squat_start_time': 0, 'squat_end_time': 0
    }


def read_frame(cap):
    return cap.read()


def process_frame(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(frame_rgb)


def check_feet(tab_heel, tabl_foot_index, tab_shoulder):
    angle_left = int(calculate_angle(tabl_foot_index[2], tabl_foot_index[3], tab_heel[2], tab_heel[3]))
    angle_right = int(calculate_angle(tab_heel[0], tab_heel[1], tabl_foot_index[0], tabl_foot_index[1]))

    print(f"Lewa stopa jes pod kątem {angle_left} stopni.")
    print(f"Prawa stopa jes pod kątem {angle_right} stopni.")

    threshold_percentage = 10

    # Assuming each entry in tab_shoulder is a tuple (x, y)
    if len(tab_shoulder) >= 2:
        shoulders_width = math.dist(tab_shoulder[0], tab_shoulder[1])  # Calculate Euclidean distance between shoulders
        heels_width = abs(tab_heel[2] - tab_heel[0])  # Assuming this is the horizontal distance between heels

        if abs(heels_width - shoulders_width) < shoulders_width * threshold_percentage / 100:
            print("Dobrze: szerokość ramion i stóp jest w przybliżeniu taka sama.")
        else:
            print("Źle: szerokość ramion i stóp powinna być podobna.")


def calculate_angle(x1, y1, x2, y2):
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return 90 - abs(angle_deg)


def check_feet_before_start(tab_heel, tab_foot_index, fps):
    before = int(len(tab_heel) - fps / 2)
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
            return True
    return False


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


def check_squat_ended(tab_shoulder_end, tab_shoulder_started, tab_hip_end, hip_hip_started):
    if len(tab_shoulder_end) > 0:
        if (tab_shoulder_end[1] <= tab_shoulder_started[1] and
                tab_shoulder_end[3] <= tab_shoulder_started[3] and
                tab_hip_end[1] <= hip_hip_started[1] and
                tab_hip_end[3] <= hip_hip_started[3]):
            return True
    return False


def update_max_depth(results, max_depth_hip, tab_knee, tab_shoulder, w, h):
    hip_y = results.pose_landmarks.landmark[23].y * h

    if max_depth_hip < hip_y:
        max_depth_hip = hip_y

        tab_knee = [
            (int(results.pose_landmarks.landmark[25].x * w),
             int(results.pose_landmarks.landmark[25].y * h)),  # Left knee
            (int(results.pose_landmarks.landmark[26].x * w),
             int(results.pose_landmarks.landmark[26].y * h))   # Right knee
        ]

        tab_shoulder = [
            (int(results.pose_landmarks.landmark[11].x * w),
             int(results.pose_landmarks.landmark[11].y * h)),  # Left shoulder
            (int(results.pose_landmarks.landmark[12].x * w),
             int(results.pose_landmarks.landmark[12].y * h))   # Right shoulder
        ]

    return max_depth_hip, tab_knee, tab_shoulder


def print_skeleton(results, frame):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def check_knee(tab_knee, tab_shoulder):
    # Assuming each entry in the lists is a tuple (x, y)
    left_knee_x = tab_knee[0][0]
    right_knee_x = tab_knee[1][0]
    left_shoulder_x = tab_shoulder[0][0]
    right_shoulder_x = tab_shoulder[1][0]

    # Check if both knees are outside the shoulders' horizontal positions
    if left_knee_x > left_shoulder_x and right_knee_x < right_shoulder_x:
        print("Dobrze: Kolana są szersze niż ramiona")
    else:
        print("Źle: Kolana powinny być szerzej niż ramiona")


def print_tips():
    print("Wskazówki po poprawnego wykanania przysiadu:")
    print("Cały czas powinny być spięty brzuch.")
    print("Kolana powinny być szersze niż ramiona")
    print("Szerokość ramion i stóp powinna być w przybliżeniu taka sama.")
    print("Stopy powinny być skierowane na zewnątrz pod kątem około 30 stopni.")
    print("\n")


def print_summary(squat_count, start_time):
    sum_time = str(int(time.time() - start_time))
    print("\nPodsumowanie")
    print(f"Zrobiłeś {squat_count} przysiadów.")
    print(f"W {sum_time} sekund.")


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


def display_frame(frame):
    cv2.imshow('Pose Estimation', frame)


def exit_requested():
    return cv2.waitKey(1) & 0xFF == ord('q')


# if __name__ == "__main__":
#     squat_Front()
