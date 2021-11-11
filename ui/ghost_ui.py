# Ghost UI

from tkinter import *
from PIL import Image, ImageTk


class GhostUI:
    def __init__(self, canvas):
        self.canvas = canvas
        self.filename = "ui/images/red_ghost_trans.png"
        self.image = Image.open(self.filename)

    def draw(self, x, y, w, h):
        self.canvas.create_rectangle(x, y, x + 80, y + 80, fill="green")
        self.canvas.pack()

        # print("x: ", x)
        # print("y: ", y)
        # print("w: ", w)
        # print("h: ", h)
        # print(self.image)

        # self.image = self.image.resize((w, h), Image.ANTIALIAS)
        new_image = ImageTk.PhotoImage(self.image)
        print(new_image)
        # self.canvas.create_image(x, y, anchor=NW, image=new_image)
        # self.canvas.create_image(x, y, image=new_image)
        self.canvas.create_image(0, 0, anchor=NW, image=new_image)
