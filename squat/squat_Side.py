import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def squat_Side(video_path, callback):
    #video_path = '//Users/janzemlo/Inzynierka/squat_vids/side1.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()

    tabs, max_depths = initialize_tabs(), initialize_max_depths()
    squat_started, squat_ended, left = False, False, True
    squat_count = 0

    print_tips()

    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.95) as pose:
        while cap.isOpened():
            ret, frame = read_frame(cap)
            if not ret:
                break

            results = process_frame(frame, pose)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                left, heel_index = determine_side_and_heel_index(results)

                print_skeleton(results, frame)
                squat_started, squat_ended, max_depths, squat_completed = process_squat_phases(results, tabs,
                                                                                               max_depths,
                                                                                               w, h, fps, start_time,
                                                                                               squat_started,
                                                                                               squat_ended,
                                                                                               left, heel_index,
                                                                                               squat_count)

                if squat_completed:
                    squat_count += 1
                    print(f"Początek {squat_count} przysiadu w {tabs['squat_start_time']} sekundzie.")
                    print(f"Koniec {squat_count} przysiadu w {tabs['squat_end_time']} sekundzie.")
                    print_summary(max_depths, w, tabs, squat_count)
                    tabs, max_depths = initialize_tabs(), initialize_max_depths()  # Reset for next squat
                    squat_started, squat_ended = False, False

                callback(frame)
            if exit_requested():
                break

        print_summary_end(squat_count, start_time)
        cleanup(cap)



def initialize_tabs():
    return {
        'hip': [], 'heel': [], 'foot_index': [], 'knee': [],
        'shoulder': [], 'shoulder_started': [], 'shoulder_end': [],
        'hip_started': [], 'hip_end': [], 'squat_start_time': 0, 'squat_end_time': 0,
        'neck_sec': set(), 'feet_sec': set()
    }


def initialize_max_depths():
    return {'hip': 0, 'knee': 0, 'knee_x': 0, 'foot_x': 0}


def read_frame(cap):
    return cap.read()


def process_frame(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(frame_rgb)


def determine_side_and_heel_index(results):
    if results.pose_landmarks.landmark[29].y > results.pose_landmarks.landmark[30].y:
        left = True
    else:
        left = False

    heel_index = 29 if left else 30
    return left, heel_index


def process_squat_phases(results, tabs, max_depths, w, h, fps, start_time, squat_started, squat_ended, left, heel_index,
                         squat_count):
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
                    int(results.pose_landmarks.landmark[heel_index].x * w),
                    int(results.pose_landmarks.landmark[heel_index].y * h)
                ]

    elif not squat_ended:
        hip_index = 23 if left else 24
        knee_index = 25 if left else 26

        hip = results.pose_landmarks.landmark[hip_index]
        knee = results.pose_landmarks.landmark[knee_index]

        # Updating maximum depth for hip and knee
        max_depths['hip'] = max(max_depths['hip'], hip.y * h)
        max_depths['knee'] = max(max_depths['knee'], knee.y * h)
        max_depths['knee_x'] = knee.x * w

        # Track foot index for shin angle calculation
        foot_index = results.pose_landmarks.landmark[heel_index - 2]  # 27 or 28 depending on left/right
        max_depths['foot_x'] = foot_index.x * w

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

            nose = [
                int(results.pose_landmarks.landmark[0].x * w),
                int(results.pose_landmarks.landmark[0].y * h)
            ]

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
                tabs['neck_sec'].add(int(elapsed_time))

            heel = [
                int(results.pose_landmarks.landmark[heel_index].x * w),
                int(results.pose_landmarks.landmark[heel_index].y * h)
            ]

            if not check_tearing_off_feet(heel, tabs['heel_before']):
                elapsed_time = time.time() - start_time
                tabs['feet_sec'].add(int(elapsed_time))

        else:
            squat_ended = True
            squat_completed = True
            tabs['squat_end_time'] = time.time() - start_time

    return squat_started, squat_ended, max_depths, squat_completed


def check_squat_started(tab_shoulder, tab_hip, fps):
    # Check the movement over a shorter period, say a quarter of a second
    before = int(len(tab_hip) - fps // 4)  # Adjusted from fps to fps // 4
    if 0 < before < len(tab_hip):
        # Reduced thresholds for quicker detection
        threshold_percentage_hip = 1.5  # Adjust the threshold as needed
        threshold_percentage_shoulder = 1.5  # Adjust the threshold as needed

        diff_hip = tab_hip[-1][1] - tab_hip[before][1]
        diff_shoulder = tab_shoulder[-1][1] - tab_shoulder[before][1]

        if (abs(diff_hip) > abs(tab_hip[before][1]) * threshold_percentage_hip / 100 and
                abs(diff_shoulder) > abs(tab_shoulder[before][1]) * threshold_percentage_shoulder / 100):
            return True
    return False



def check_depth(max_depth_hip, max_depth_knee):
    return max_depth_hip < max_depth_knee


def check_shins(max_depth_knee_x, max_depth_foot_x, w):
    threshold_percentage = w * 1.5 / 100
    return abs(max_depth_knee_x - max_depth_foot_x) < threshold_percentage


def check_squat_ended(tab_shoulder_end, tab_shoulder_started, tab_hip_end, hip_hip_started):
    if len(tab_shoulder_end) > 0:
        if (tab_shoulder_end[1] <= tab_shoulder_started[1] and
                tab_shoulder_end[3] <= tab_shoulder_started[3] and
                tab_hip_end[1] <= hip_hip_started[1] and
                tab_hip_end[3] <= hip_hip_started[3]):
            return True
    return False


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


def check_side_left(results):
    if results.pose_landmarks.landmark[29].y > results.pose_landmarks.landmark[30].y:
        return True
    return False


def check_neck(shoulder, hip, nose):
    m = (hip[1] - shoulder[1]) / (hip[0] - shoulder[0])
    b = shoulder[1] - m * shoulder[0]

    y_nose = m * nose[0] + b

    if min(shoulder[1], hip[1]) <= y_nose <= max(shoulder[1], hip[1]):
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


def print_skeleton(results, frame):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def display_frame(frame):
    cv2.imshow('Pose Estimation', frame)


def exit_requested():
    return cv2.waitKey(1) & 0xFF == ord('q')


def print_tips():
    print("Wskazówki po poprawnego wykanania przysiadu:")
    print("Cały czas powinny być spięty brzuch.")
    print("Nie powinno się się odrywać stóp podczas wykonywania ćwiczenia.")
    print("Poprawna głębokość przysiadu jest wtedy, kiedy wykość koland i bioder jest taka sama.")
    print("Piszczele powinny być pod kątem 90 stopni względem ziemi na końcu przysiadu.")
    print("\n")


def print_summary_end(squat_count, start_time):
    sum_time = str(int(time.time() - start_time))
    print("\nPodsumowanie")
    print(f"Zrobiłeś {squat_count} przysiadów.")
    print(f"W {sum_time} sekund.")

def print_summary(max_depths, w, tabs, squat_count):
    depth_correct = check_depth(max_depths['hip'], max_depths['knee'])
    shins_correct = check_shins(max_depths['knee_x'], max_depths['foot_x'], w)

    depth_message = "Dobrze: poprawna głębokość przysiadu." if depth_correct else "Źle: niepoprawna głębokość przysiadu."
    shins_message = "Dobrze: piszczele są pod kątem 90 stopni względem ziemi." if shins_correct else "Źle: piszczele nie są pod kątem 90 stopni względem ziemi."

    print(f"{depth_message} podczas {squat_count} przysiadów.")
    print(f"{shins_message} podczas {squat_count} przysiadów")

    if tabs['neck_sec']:
        print(
            f"Nieprawidłowe ułożenie szyi wystąpiło w sekundach: {tabs['neck_sec']} "
            f"podczas {squat_count} przysiadów.")
    else:
        print(f"Prawidłowe ułożenie szyi wystąpiło podczas wykonywania każdego przysiadu.")

    if tabs['feet_sec']:
        print(f"Stopy uniosły się w sekund: {tabs['feet_sec']} podczas {squat_count} przysiadów.")
    else:
        print(f"Nie wykryto podnoszenia stóp podczas wykonywania podczas podczas wykonywania każdego przysiadu.")

    print('\n')


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     squat_Side()
