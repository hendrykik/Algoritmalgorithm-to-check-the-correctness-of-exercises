from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog
from squat.squat_Front import squat_Front
from squat.squat_Side import squat_Side
from deadlift.deadlift_Front import deadlift_Front
from deadlift.deadlift_Side import deadlift_Side

def main():
    window = tk.Tk()
    window.title("Exercise Analysis")
    window.geometry("1280x720")

    canvas = tk.Canvas(window, width=1280, height=720)
    canvas.pack()

    def update_gui(frame):
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        photo = ImageTk.PhotoImage(image=pil_image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo

    def run_video(file_path, analysis_function):
        cap = cv2.VideoCapture(file_path)
        def process_frame():
            ret, frame = cap.read()
            if ret:
                processed_frame = analysis_function(frame)
                update_gui(processed_frame)
                window.after(10, process_frame)
            else:
                cap.release()
        process_frame()

    def choose_file(analysis_function):
        file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            run_video(file_path, analysis_function)

    def on_squat_front():
        choose_file(squat_Front)

    def on_squat_side():
        choose_file(squat_Side)

    btn_squat_front = tk.Button(window, text="Squat Front Analysis", command=on_squat_front)
    btn_squat_front.pack(pady=10)

    btn_squat_side = tk.Button(window, text="Squat Side Analysis", command=on_squat_side)
    btn_squat_side.pack(pady=10)

    def on_deadlift_front():
        choose_file(deadlift_Front)

    def on_deadlift_side():
        choose_file(deadlift_Side)

    btn_deadlift_front = tk.Button(window, text="Deadlift Front Analysis", command=on_deadlift_front)
    btn_deadlift_front.pack(pady=10)

    btn_deadlift_side = tk.Button(window, text="Deadlift Side Analysis", command=on_deadlift_side)
    btn_deadlift_side.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    main()
