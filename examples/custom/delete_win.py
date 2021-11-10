# https://stackoverflow.com/questions/61988801/exit-program-within-a-tkinter-class

# This GUI class exits correctly

# Quickstart to see output unbuffered:
#   python -u delete_win.py

import tkinter as tk
from tkinter import *


class GUI(Tk):
    def __init__(self):
        super().__init__()

        # Close the app when the window's X is pressed
        self.protocol("WM_DELETE_WINDOW", self.closing)

        # When this var is set to 1, the move function can continue
        self.var = tk.IntVar()

        # Close the app if the button is pressed
        button = tk.Button(self, text="Exit",
                           command=self.destroy)
        button.place(relx=.5, rely=.5, anchor="c")

        # Step forward
        self.step_button = tk.Button(self, text="Step",
                                     command=lambda: self.var.set(1))
        self.step_button.place(relx=.5, rely=.75, anchor="c")

    def move(self):
        print("doing stuff")  # simulates stuff being done
        self.step_button.wait_variable(self.var)
        self.after(0, self.move)

    def closing(self):
        self.destroy()
        self.var.set("")
        self.step_button.wait_variable()


app = GUI()
app.move()
app.mainloop()
