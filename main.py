import tkinter as tk
from PIL import Image, ImageTk
from squat.squat_Front import squat_Front
from squat.squat_Side import squat_Side
from deadlift.deadlift_Front import deadlift_Front
from deadlift.deadlift_Side import deadlift_Side
import threading
import cv2
from tkinter import filedialog, font
import sys
from tkmacosx import Button

class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)

def set_background_image(frame, image_path):
    # Załadowanie obrazka
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((1280, 720), Image.ANTIALIAS)  # Dostosowanie rozmiaru do okna
    bg_photo = ImageTk.PhotoImage(bg_image)

    # Ustawienie obrazka jako tła
    bg_label = tk.Label(frame, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.image = bg_photo

def main():
    bg_color = "#f0f0f0"  # Jasny szary


    window = tk.Tk()
    window.title("Analiza ćwiczeń")
    window.geometry("1280x720")
    window.configure(bg=bg_color)

    display_frame = tk.Frame(window)
    display_frame.pack(fill=tk.BOTH, expand=True)
    display_frame.config(bg='#c3573c')


    canvas = tk.Canvas(display_frame, width=640, height=360)
    canvas.pack(side=tk.LEFT, padx=20, pady=20)
    canvas.configure(bg="#383838")
    alg_number = 0
    global thread, thread_stop_flag, choosed_camera
    thread = None
    thread_stop_flag = threading.Event()
    choosed_camera = None

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
    set_background_image(main_frame, 'back.png')
    main_frame.grid_columnconfigure(0, weight=1)  # Left buffer column
    main_frame.grid_columnconfigure(1, weight=2)  # Squat buttons column
    main_frame.grid_columnconfigure(2, weight=2)  # Deadlift buttons column
    main_frame.grid_columnconfigure(3, weight=1)

    choose_camera_frame = tk.Frame(window)
    choose_camera_frame.config(bg='#c3573c')
    choose_camera_frame.grid_columnconfigure(0, weight=1)  # Left buffer column
    choose_camera_frame.grid_columnconfigure(1, weight=1)  # Squat buttons column

    for frame in (main_frame, display_frame, choose_camera_frame):
        frame.place(x=0, y=0, width=1280, height=720)

    def on_squat_side():
        global alg_number, thread
        alg_number = 0
        dislay_choose_cam()

    def on_squat_front():
        global alg_number, thread
        alg_number = 1
        dislay_choose_cam()

    def on_deadlift_side():
        global alg_number, thread
        alg_number = 2
        dislay_choose_cam()

    def on_deadlift_front():
        global alg_number, thread
        alg_number = 3
        dislay_choose_cam()
    
    def on_back_to_main():
        global thread, thread_stop_flag
        thread_stop_flag.set()
        output_text.delete("1.0", tk.END)
        show_frame(main_frame)
    
    def on_choosed_file():
        global choosed_camera
        choosed_camera = choose_file()
        thread = threading.Thread(target=run_algorithm)
        thread.start()

    def on_cam_choosed():
        global choosed_camera
        choosed_camera = 1
        thread = threading.Thread(target=run_algorithm)
        thread.start()

    def dislay_choose_cam():
        show_frame(choose_camera_frame) 


    def run_algorithm():
        global alg_number, thread_stop_flag, choosed_camera      
        try:
            if choosed_camera != None:
                show_frame(display_frame)
                if alg_number == 0:
                    squat_Side(choosed_camera, update_image)
                elif alg_number == 1:
                    squat_Front(choosed_camera, update_image)
                elif alg_number == 2:
                    deadlift_Side(choosed_camera, update_image)
                elif alg_number == 3:
                    deadlift_Front(choosed_camera, update_image)
                else:
                    print("Błąd z algorytmem.") 
                while not thread_stop_flag.is_set(): 
                    # Add necessary checks or short sleeps if needed
                    pass
        except Exception as e:
            print("Błąd w wątku algorytmu", e)

    
    def show_frame(frame):
        frame.tkraise()
    
    def choose_file():
        file_path = filedialog.askopenfilename(title="Wybierz plik viedo", filetypes=[("Pliki video", "*.mp4 *.avi")])
        if file_path:
            return file_path
        
    def update_output(message):
        output_text.insert(tk.END, message + "\n")
        output_text.see(tk.END)
        output_text.configure(bg="#ffffff", fg="#000000")
    
    custom_font = font.Font(family="Helvetica", size=12, weight="bold")

    btn_squat_side = tk.Button(main_frame, text="Analiza przysiadu\nWidok z boku", 
                           command=on_squat_side, bg='red', fg="black", 
                           font=custom_font, relief="raised", bd=3, highlightbackground='pink')
    btn_squat_side.grid(row=3, column=0, padx=20, pady=120)

    btn_squat_front = tk.Button(main_frame, text="Analiza przysiadu\nWidok z przodu", 
                                command=on_squat_front, bg="#FFA07A", fg="black",  
                                font=custom_font, relief="raised", bd=3, highlightbackground='pink') 
    btn_squat_front.grid(row=4, column=0, padx=20, pady=160)

    btn_deadlift_side = tk.Button(main_frame, text="Analiza martwego ciągu\nWidok z boku", 
                                command=on_deadlift_side, bg="#FFD700", fg="black", 
                                font=custom_font, relief="raised", bd=3, highlightbackground='pink')
    btn_deadlift_side.grid(row=3, column=3, padx=20, pady=120)

    btn_deadlift_front = tk.Button(main_frame, text="Analiza martwego ciągu\nWidok z przodu", 
                                command=on_deadlift_front, bg="#FFD700", fg="black", 
                                font=custom_font, relief="raised", bd=3, highlightbackground='pink')
    btn_deadlift_front.grid(row=4, column=3, padx=20, pady=160)

    btn_choose_camera_file = tk.Button(choose_camera_frame, text="Video z pliku", 
                                command=on_choosed_file, bg="#FFD700", fg="black",  
                                font=custom_font, relief="raised", bd=3, highlightbackground='pink')
    btn_choose_camera_file.grid(row=0, column=0, padx=0, pady=200)

    btn_choose_camera_cam = tk.Button(choose_camera_frame, text="Video z kamery z komputera", 
                                command=on_cam_choosed, bg="#FFD700", fg="black", 
                                font=custom_font, relief="raised", bd=3, highlightbackground='pink')
    btn_choose_camera_cam.grid(row=0, column=1, padx=0, pady=0)

    output_text = tk.Text(display_frame, height=27, width=60, wrap=tk.WORD)
    output_text.pack(pady=(180, 0))  # Ustawia pady na 180 pikseli od góry i 0 od dołu dla output_text

    back_button = tk.Button(display_frame, text="Powrót do ekranu głównego", command=on_back_to_main,
                            font=custom_font, relief="raised", bd=3, highlightbackground='pink')
    # Ustawia pady na 80 pikseli od góry dla back_button, co daje odstęp 80 pikseli między output_text a back_button
    back_button.pack(pady=(80, 0))  

    sys.stdout = TextRedirector(output_text)
    show_frame(main_frame)
    window.mainloop()

if __name__ == "__main__":
    main()