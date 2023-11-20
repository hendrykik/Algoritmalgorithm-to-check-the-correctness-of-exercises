import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def deadlift_Side(video_path, callback):
    # video_path = '//Users/janzemlo/Inzynierka/deadlift_vids/side1.mp4'
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
        'eye_started': [], 'eye_end': [], 'eye': [],
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
        # if not check_feet_before_start(tabs['heel'], tabs['foot_index'], fps):
        #     tabs['heel'].append((
        #         int(results.pose_landmarks.landmark[29].x * w),
        #         int(results.pose_landmarks.landmark[29].y * h),
        #         int(results.pose_landmarks.landmark[30].x * w),
        #         int(results.pose_landmarks.landmark[30].y * h)
        #     ))

        #     tabs['foot_index'].append((
        #         int(results.pose_landmarks.landmark[31].x * w),
        #         int(results.pose_landmarks.landmark[31].y * h),
        #         int(results.pose_landmarks.landmark[32].x * w),
        #         int(results.pose_landmarks.landmark[32].y * h)
        #     ))
       # else:
        if not check_deadlift_started(tabs['shoulder'], tabs['eye'], fps):
            tabs['shoulder'].append((
                int(results.pose_landmarks.landmark[11].x * w),
                int(results.pose_landmarks.landmark[11].y * h),
                int(results.pose_landmarks.landmark[12].x * w),
                int(results.pose_landmarks.landmark[12].y * h)
            ))
            tabs['eye'].append((
                int(results.pose_landmarks.landmark[2].x * w),
                int(results.pose_landmarks.landmark[2].y * h),
                int(results.pose_landmarks.landmark[5].x * w),
                int(results.pose_landmarks.landmark[5].y * h)
            ))
        else:
            deadlift_started = True
            # print("started")
            tabs['deadlift_start_time'] = time.time() - start_time  # Record the start time of the deadlift
            tabs['shoulder_started'] = [
                int(results.pose_landmarks.landmark[11].x * w),
                int(results.pose_landmarks.landmark[11].y * h),
                int(results.pose_landmarks.landmark[12].x * w),
                int(results.pose_landmarks.landmark[12].y * h)
            ]
            tabs['eye_started'] = [
                int(results.pose_landmarks.landmark[2].x * w),
                int(results.pose_landmarks.landmark[2].y * h),
                int(results.pose_landmarks.landmark[5].x * w),
                int(results.pose_landmarks.landmark[5].y * h)
            ]

    elif not deadlift_ended:
        if not check_deadlift_ended(tabs['shoulder_end'], tabs['shoulder_started'],
                                 tabs['eye_end'], tabs['eye_started']):
            tabs['shoulder_end'] = [
                int(results.pose_landmarks.landmark[11].x * w),
                int(results.pose_landmarks.landmark[11].y * h),
                int(results.pose_landmarks.landmark[12].x * w),
                int(results.pose_landmarks.landmark[12].y * h)
            ]
    
            tabs['eye_end'] = [
                int(results.pose_landmarks.landmark[2].x * w),
                int(results.pose_landmarks.landmark[2].y * h),
                int(results.pose_landmarks.landmark[5].x * w),
                int(results.pose_landmarks.landmark[5].y * h)
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


def check_deadlift_started(tab_shoulder, tab_eye, fps):
    before = int(len(tab_eye) - fps)
    if len(tab_shoulder) > fps and len(tab_eye) > fps:
        #before = -fps  # Index for the previous second

        # Calculate the vertical movement of the shoulders
        diff_shoulder = tab_shoulder[-1][1] - tab_shoulder[before][1]

        # Check if eyes and shoulders are at similar height initially
        similar_height = abs(tab_eye[before][1] - tab_shoulder[before][1]) < tab_shoulder[-1][1] * 0.2  # 20 is a threshold, adjust as needed

        # Check if shoulders move upwards quickly
        # Negative diff indicates an upward movement
        if similar_height:
            if -diff_shoulder > 20:  # 20 is a threshold for quick upward movement, adjust as needed
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

def check_deadlift_ended(tab_shoulder_end, tab_shoulder_started, tab_eye_end, tab_eye_started):
    if len(tab_shoulder_end) > 0:
        if (tab_shoulder_end[1] >= tab_shoulder_started[1] and
                tab_shoulder_end[3] >= tab_shoulder_started[3] and
                tab_eye_end[1] >= tab_eye_started[1] and
                tab_eye_end[3] >= tab_eye_started[3]):
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


# if __name__ == "__main__":
#     deadlift_Side()
