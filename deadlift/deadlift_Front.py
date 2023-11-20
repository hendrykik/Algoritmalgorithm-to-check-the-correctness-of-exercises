import cv2
import mediapipe as mp
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def deadlift_Front(video_path, callback):
    # video_path = '//Users/janzemlo/Inzynierka/deadlift_vids/front1.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = time.time()

    tabs = initialize_tabs()
    deadlift_count = 0
    deadlift_started, deadlift_ended, deadlift_completed = False, False, False

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = read_frame(cap)
            if not ret:
                break

            results = process_frame(frame, pose)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                deadlift_started, deadlift_ended, deadlift_completed = \
                    process_deadlift_phases(results, tabs, fps, w, h, start_time,
                                            deadlift_started, deadlift_ended, deadlift_count)

                if deadlift_completed:
                    deadlift_count += 1
                    print(f"Start of {deadlift_count} deadlift at {tabs['deadlift_start_time']} seconds")
                    print(f"End of {deadlift_count} deadlift at {tabs['deadlift_end_time']} seconds")
                    tabs = initialize_tabs()
                    deadlift_started, deadlift_ended = False, False

                print_skeleton(results, frame)

            callback(frame)
            if exit_requested():
                break

        cleanup(cap)


def initialize_tabs():
    return {
        'hip': [], 'heel': [], 'foot_index': [], 'knee': [],
        'shoulder': [], 'shoulder_started': [], 'shoulder_end': [],
        'hip_started': [], 'hip_end': [],
        'deadlift_start_time': 0, 'deadlift_end_time': 0
    }


def read_frame(cap):
    return cap.read()


def process_frame(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(frame_rgb)


def process_deadlift_phases(results, tabs, fps, w, h, start_time, deadlift_started, deadlift_ended, deadlift_count):
    deadlift_completed = False

    if not deadlift_started:
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
            if not check_deadlift_started(tabs['shoulder'], tabs['hip'], fps):
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
                deadlift_started = True
                check_feet(tabs['heel'][-1], tabs['foot_index'][-1])
                tabs['deadlift_start_time'] = time.time() - start_time  # Record the start time of the deadlift
                tabs['shoulder_started'] = [
                    int(results.pose_landmarks.landmark[11].x * w),
                    int(results.pose_landmarks.landmark[11].y * h),
                    int(results.pose_landmarks.landmark[12].x * w),
                    int(results.pose_landmarks.landmark[12].y * h)
                ]
                tabs['hip_started'] = [
                    int(results.pose_landmarks.landmark[23].x * w),
                    int(results.pose_landmarks.landmark[23].y * h),
                    int(results.pose_landmarks.landmark[24].x * w),
                    int(results.pose_landmarks.landmark[24].y * h)
                ]
                tabs['heel_before'] = [
                    int(results.pose_landmarks.landmark[29].x * w),
                    int(results.pose_landmarks.landmark[29].y * h),
                    int(results.pose_landmarks.landmark[30].x * w),
                    int(results.pose_landmarks.landmark[30].y * h)
                ]

    elif not deadlift_ended:
        if not check_deadlift_ended(tabs['shoulder_end'], tabs['shoulder_started'],
                                    tabs['hip_end'], tabs['hip_started']):
            tabs['shoulder_end'] = [
                int(results.pose_landmarks.landmark[11].x * w),
                int(results.pose_landmarks.landmark[11].y * h),
                int(results.pose_landmarks.landmark[12].x * w),
                int(results.pose_landmarks.landmark[12].y * h)
            ]

            tabs['hip_end'] = [
                int(results.pose_landmarks.landmark[23].x * w),
                int(results.pose_landmarks.landmark[23].y * h),
                int(results.pose_landmarks.landmark[24].x * w),
                int(results.pose_landmarks.landmark[24].y * h)
            ]
        else:
            deadlift_ended = True
            deadlift_completed = True
            tabs['deadlift_end_time'] = time.time() - start_time

    elif deadlift_completed:
        # Print the start and end time for the current deadlift
        print(f"Start of {deadlift_count + 1} deadlift at {tabs['deadlift_start_time']} seconds")
        print(f"End of {deadlift_count + 1} deadlift at {tabs['deadlift_end_time']} seconds")

        # Reset variables for next deadlift
        deadlift_started = False
        deadlift_ended = False
        tabs = initialize_tabs()
        tabs['deadlift_start_time'] = 0
        tabs['deadlift_end_time'] = 0
        deadlift_count += 1

    return deadlift_started, deadlift_ended, deadlift_completed


def check_deadlift_started(tab_shoulder, tab_hip, fps):
    before = int(len(tab_hip) - fps)
    if before > 0:
        threshold_percentage_hip = 3
        threshold_percentage_shoulder = 3

        diff_hip = tab_hip[-1][1] - tab_hip[before][1]
        diff_shoulder = tab_shoulder[-1][1] - tab_shoulder[before][1]

        if (
                -diff_hip > abs(tab_hip[before][1]) * threshold_percentage_hip / 100 and
                -diff_shoulder > abs(tab_shoulder[before][1]) * threshold_percentage_shoulder / 100
        ):
            return True
    return False


def check_deadlift_ended(tab_shoulder_end, tab_shoulder_started, tab_hip_end, hip_hip_started):
    if len(tab_shoulder_end) > 0:
        if (tab_shoulder_end[1] >= tab_shoulder_started[1] and
                tab_shoulder_end[3] >= tab_shoulder_started[3] and
                tab_hip_end[1] >= hip_hip_started[1] and
                tab_hip_end[3] >= hip_hip_started[3]):
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


def print_skeleton(results, frame):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


def display_frame(frame):
    cv2.imshow('Pose Estimation', frame)


def exit_requested():
    return cv2.waitKey(1) & 0xFF == ord('q')

def check_feet(tab_heel, tabl_foot_index):
    angle_left = int(calculate_angle(tabl_foot_index[2], tabl_foot_index[3], tab_heel[2], tab_heel[3]))
    angle_right = int(calculate_angle(tab_heel[0], tab_heel[1], tabl_foot_index[0], tabl_foot_index[1]))

    print("Setup with your toes pointing about 15° out")
    print(f"Left foot angle is {angle_left} degrees")
    print(f"Right foot angle is {angle_right} degrees")


def calculate_angle(x1, y1, x2, y2):
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    return 90 - abs(angle_deg)


# if __name__ == "__main__":
#     deadlift_Front()
