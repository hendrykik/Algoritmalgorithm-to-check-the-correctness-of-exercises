from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog
import threading
# Import your algorithm functions
from squat.squat_Front import squat_Front
from squat.squat_Side import squat_Side
from deadlift.deadlift_Front import deadlift_Front
from deadlift.deadlift_Side import deadlift_Side

def main():
    window = tk.Tk()
    window.title("Exercise Analysis")
    window.geometry("1280x720")

    # Create a canvas for video frame display
    canvas = tk.Canvas(window, width=1280, height=720)
    canvas.pack()

    def update_gui(frame):
        # Convert the OpenCV image to a format Tkinter can use
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        photo = ImageTk.PhotoImage(image=pil_image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep a reference to avoid garbage collection

    def show_frame(frame):
        frame.tkraise()

    main_frame = tk.Frame(window)
    squat_frame = tk.Frame(window)
    deadlift_frame = tk.Frame(window)

    for frame in (main_frame, squat_frame, deadlift_frame):
        frame.place(x=0, y=0, width=1280, height=720)

    def on_squat():
        show_frame(squat_frame)

    def on_deadlift():
        show_frame(deadlift_frame)

    # Create view selection frame
    def create_view_frame(exercise_type, frame):
        def on_side():
            choose_file(f"{exercise_type}_Side")

        def on_front():
            choose_file(f"{exercise_type}_Front")

        btn_side = tk.Button(frame, text="Side View", command=on_side)
        btn_side.pack(pady=10)

        btn_front = tk.Button(frame, text="Front View", command=on_front)
        btn_front.pack(pady=10)

        btn_back = tk.Button(frame, text="Back", command=lambda: show_frame(main_frame))
        btn_back.pack(pady=10)

    create_view_frame("squat", squat_frame)
    create_view_frame("deadlift", deadlift_frame)

    def choose_file(algorithm_name):
        file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            run_algorithm(algorithm_name, file_path)

    def run_algorithm(algorithm_name, file_path):
        print(f"Running {algorithm_name} on {file_path}")
        if algorithm_name == "squat_Front":
            squat_Front(file_path, update_gui)
        elif algorithm_name == "squat_Side":
            squat_Side(file_path, update_gui)
        elif algorithm_name == "deadlift_Front":
            deadlift_Front(file_path, update_gui)
        elif algorithm_name == "deadlift_Side":
            deadlift_Side(file_path, update_gui)

    btn_squat = tk.Button(main_frame, text="Analysis of Squat", command=on_squat)
    btn_squat.pack(pady=10)

    btn_deadlift = tk.Button(main_frame, text="Analysis of Deadlift", command=on_deadlift)
    btn_deadlift.pack(pady=10)

    show_frame(main_frame)
    window.mainloop()

if __name__ == "__main__":
    main()
