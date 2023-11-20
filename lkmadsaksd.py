import tkinter as tk
from PIL import Image, ImageTk
from squat.squat_Front import squat_Front
import threading
import cv2

def main():
    window = tk.Tk()
    window.title("Squat Analysis")
    window.geometry("1280x720")

    canvas = tk.Canvas(window, width=1280, height=720)
    canvas.pack()

    def update_image(frame):
        # Convert the OpenCV image to a format Tkinter can use
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image)
        photo = ImageTk.PhotoImage(image=pil_image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo  # Keep a reference to avoid garbage collection

    def run_squat_front():
        squat_Front("", update_image)

    # Run the squat analysis in a separate thread
    thread = threading.Thread(target=run_squat_front)
    thread.start()

    window.mainloop()

if __name__ == "__main__":
    main()
