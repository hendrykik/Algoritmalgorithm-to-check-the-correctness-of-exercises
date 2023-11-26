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
    deadlift_started, deadlift_ended, deadlift_completed, deadlift_max = False, False, False, False

    print_tips()

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = read_frame(cap)
            if not ret:
                break

            results = process_frame(frame, pose)

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                deadlift_started, deadlift_ended, deadlift_completed, deadlift_max = \
                    process_deadlift_phases(results, tabs, fps, w, h, start_time,
                                            deadlift_started, deadlift_ended, deadlift_count, deadlift_max)

                if deadlift_completed:
                    deadlift_count += 1
                    print(f"Początek  {deadlift_count} martwego ciągu w {tabs['deadlift_start_time']} sekundzie.")
                    print(f"Koniec {deadlift_count} martwego ciągu w {tabs['deadlift_end_time']} sekundzie.")
                    tabs = initialize_tabs()
                    deadlift_started, deadlift_ended, deadlift_max = False, False, False

                print_skeleton(results, frame)

            callback(frame)
            if exit_requested():
                break
        
        print_summary(deadlift_count, start_time)
        cleanup(cap)


def initialize_tabs():
    return {
        'hip': [], 'heel': [], 'foot_index': [], 'knee': [],
        'shoulder': [], 'shoulder_started': [], 'shoulder_end': [],
        'eye_started': [], 'eye_end': [], 'eye': [], 'wrist_started': [],
        'deadlift_start_time': 0, 'deadlift_end_time': 0
    }


def read_frame(cap):
    return cap.read()


def process_frame(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(frame_rgb)


def process_deadlift_phases(results, tabs, fps, w, h, start_time, deadlift_started, deadlift_ended, deadlift_count, deadlift_max):
    deadlift_completed = False

    if not deadlift_started:
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
            tabs['wrist_started'] = [
                int(results.pose_landmarks.landmark[15].x * w),
                int(results.pose_landmarks.landmark[15].y * h),
                int(results.pose_landmarks.landmark[16].x * w),
                int(results.pose_landmarks.landmark[16].y * h)
            ]
            check_shoulder_start(tabs['shoulder_started'], tabs['wrist_started'])

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

        
            if not deadlift_max:
                if not check_deadlift_max(tabs['shoulder'], fps):
                    tabs['shoulder'].append((
                        int(results.pose_landmarks.landmark[11].x * w),
                        int(results.pose_landmarks.landmark[11].y * h),
                        int(results.pose_landmarks.landmark[12].x * w),
                        int(results.pose_landmarks.landmark[12].y * h)
                    ))
                    tabs['hip'] = [
                        int(results.pose_landmarks.landmark[11].x * w),
                        int(results.pose_landmarks.landmark[11].y * h),
                        int(results.pose_landmarks.landmark[12].x * w),
                        int(results.pose_landmarks.landmark[12].y * h)
                    ]
                else:
                    check_shoulder_at_max(tabs['shoulder'][-1], tabs['hip'])
                    deadlift_max = True

        else:
            deadlift_ended = True
            deadlift_completed = True
            tabs['deadlift_end_time'] = time.time() - start_time
    

    return deadlift_started, deadlift_ended, deadlift_completed, deadlift_max


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


def check_shoulder_start(tab_shoulder_started, tab_wrist_started):
    is_facing_left = tab_shoulder_started[0] < tab_shoulder_started[2]

    if is_facing_left:
        # Check if both shoulders are ahead of the wrists when facing left
        if tab_shoulder_started[0] < tab_wrist_started[0] and tab_shoulder_started[2] < tab_wrist_started[2]:
            print("Dobrze: barki są przed sztangą.")
        else:
            print("Źle: barki nie są przed sztangą.")
    else:
        # Check if both shoulders are ahead of the wrists when facing right
        if tab_shoulder_started[0] > tab_wrist_started[0] and tab_shoulder_started[2] > tab_wrist_started[2]:
            print("Dobrze: barki są przed sztangą.")
        else:
            print("Źle: barki nie są przed sztangą.")


def check_deadlift_max(tab_shoulder, fps):
    before = int(len(tab_shoulder) - fps / 2)
    if before > 0:
        threshold_percentage = 2
        # Compare the current Y value of the shoulders to the previous value
        diff_shoulder_y = tab_shoulder[-1][1] - tab_shoulder[before][1]

        # Check if the Y value has increased beyond a threshold
        if diff_shoulder_y > abs(tab_shoulder[before][1]) * threshold_percentage / 100:
            return True
    return False


def check_shoulder_at_max(tab_shoulder, tab_hip):
    is_facing_left = tab_shoulder[0] < tab_shoulder[2]

    if is_facing_left:
        # Check if left hip is ahead of the left shoulder
        if tab_hip[0] < tab_shoulder[0]:
            print("Źle: biodra są przed barkami.")
        else:
            print("Dobrze: biodra nie są przed barkami.")
    else:
        # Check if right hip is ahead of the right shoulder
        if tab_hip[2] > tab_shoulder[2]:
            print("Źle: biodra są przed barkami.")
        else:
            print("Dobrze: biodra nie są przed barkami.")


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


def check_shoulders_deflection():
    return 1


def print_tips():
    print("Wskazówki po poprawnego wykanania martwego ciągu:")
    print("Cały czas powinny być spięty brzuch.")
    print("Nie powinno się wygiąć do tyłu przy końcu ćwiczenia.")
    print("Podczas startu ćwiczenia barki powinny być przed sztangą.")
    print("\n")


def print_summary(deadlift_count, start_time):
    sum_time = str(int(time.time() - start_time))
    print("\nPodsumowanie")
    print(f"Zrobiłeś {deadlift_count} martwych ciągów.")
    print(f"W {sum_time} sekund.")


# if __name__ == "__main__":
#     deadlift_Side()
