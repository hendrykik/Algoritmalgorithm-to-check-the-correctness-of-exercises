import tkinter as tk
from PIL import Image, ImageTk
from squat.squat_Front import squat_Front
from squat.squat_Side import squat_Side
from deadlift.deadlift_Front import deadlift_Front
from deadlift.deadlift_Side import deadlift_Side
import threading
import cv2
from tkinter import filedialog
import sys

class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)

def main():
    window = tk.Tk()
    window.title("Squat Analysis")
    window.geometry("1280x720")

    display_frame = tk.Frame(window)
    display_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(display_frame, width=640, height=360)
    canvas.pack(side=tk.LEFT, padx=20, pady=20)
    alg_number = 0
    global thread, thread_stop_flag
    thread = None
    thread_stop_flag = threading.Event()

    def update_image(frame):
        target_width = 640  # Szerokość docelowa dla obrazu
        target_height = 360  # Wysokość docelowa dla obrazu

        # Konwersja obrazu z formatu OpenCV do PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Przeskalowanie obrazu do docelowego rozmiaru
        image = image.resize((target_width, target_height), Image.ANTIALIAS)

        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Zachowanie referencji do obiektu PhotoImage

    main_frame = tk.Frame(window)

    for frame in (main_frame, display_frame):
        frame.place(x=0, y=0, width=1280, height=720)

    def on_squat_side():
        global alg_number, thread
        alg_number = 0
        thread = threading.Thread(target=run_algorithm)
        thread.start()

    def on_squat_front():
        global alg_number, thread
        alg_number = 1
        thread = threading.Thread(target=run_algorithm)
        thread.start()

    def on_deadlift_side():
        global alg_number, thread
        alg_number = 2
        thread = threading.Thread(target=run_algorithm)
        thread.start()

    def on_deadlift_front():
        global alg_number, thread
        alg_number = 3
        thread = threading.Thread(target=run_algorithm)
        thread.start()
    
    def on_back_to_main():
        global thread, thread_stop_flag
        thread_stop_flag.set()
        show_frame(main_frame)

    def run_algorithm():
        global alg_number, thread_stop_flag        
        try:
            path = choose_file()
            show_frame(display_frame)
            if path:
                if alg_number == 0:
                    squat_Side(path, update_image)
                elif alg_number == 1:
                    squat_Front(path, update_image)
                elif alg_number == 2:
                    deadlift_Side(path, update_image)
                elif alg_number == 3:
                    deadlift_Front(path, update_image)
                else:
                    print("Error, alogithm do not exist.") 
                while not thread_stop_flag.is_set():
                    # Add necessary checks or short sleeps if needed
                    pass
        except Exception as e:
            print("Error in algorithm thread:", e)

    
    def show_frame(frame):
        frame.tkraise()
    
    def choose_file():
        file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video files", "*.mp4 *.avi")])
        if file_path:
            return file_path
        
    def update_output(message):
        output_text.insert(tk.END, message + "\n")
        output_text.see(tk.END)
    
    btn_squat_side = tk.Button(main_frame, text="Analysis of Squat\n Side view", command=on_squat_side)
    btn_squat_side.pack(pady=60)

    btn_squat_front = tk.Button(main_frame, text="Analysis of Squat\n Front view", command=on_squat_front)
    btn_squat_front.pack(pady=60)

    btn_deadlift_side = tk.Button(main_frame, text="Analysis of Deadlift\n Side view", command=on_deadlift_side)
    btn_deadlift_side.pack(pady=60)

    btn_deadlift_front = tk.Button(main_frame, text="Analysis of Deadlift\n Front view", command=on_deadlift_front)
    btn_deadlift_front.pack(pady=60)

    output_text = tk.Text(display_frame, height=25, width=50)
    output_text.pack(pady=100)

    back_button = tk.Button(display_frame, text="Back to Main Menu", command=on_back_to_main)
    back_button.pack(pady=50)

    sys.stdout = TextRedirector(output_text)

    show_frame(main_frame)
    window.mainloop()

if __name__ == "__main__":
    main()