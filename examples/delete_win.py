# https://stackoverflow.com/questions/61988801/exit-program-within-a-tkinter-class

from tkinter import *


class GUI(Tk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.closing)

    def closing(self):
        self.destroy()


app = GUI()
app.mainloop()
