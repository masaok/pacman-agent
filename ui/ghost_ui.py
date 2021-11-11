# Ghost UI

from tkinter import *
from PIL import Image, ImageTk


class GhostUI:
    def __init__(self, canvas):
        self.canvas = canvas
        self.filename = "ui/images/red_ghost_trans.png"
        self.image = Image.open(self.filename)

    def draw(self, x, y, w, h):
        # self.canvas.create_rectangle(x, y, x + w, y + h, fill="pink")
        margin = 10
        self.canvas.create_oval(
            [x + margin, y + margin, x + w - margin, y + h - margin],
            fill="pink")

        # Create Text: https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/create_text.html
        # Anchors: https://www.tutorialspoint.com/python/tk_anchors.htm
        font = "Arial 30 bold"
        self.canvas.create_text(x + 0.5 * w, y + 0.5 * h, anchor=CENTER,
                                font=font, text="G", fill="red")

        # TODO: None of this image stuff works yet
        # self.canvas.pack()

        # print("x: ", x)
        # print("y: ", y)
        # print("w: ", w)
        # print("h: ", h)
        # print(self.image)

        # self.image = self.image.resize((w, h), Image.ANTIALIAS)
        # new_image = ImageTk.PhotoImage(self.image)
        # print(new_image)
        # self.canvas.create_image(x, y, anchor=NW, image=new_image)
        # self.canvas.create_image(x, y, image=new_image)
        # self.canvas.create_image(0, 0, anchor=NW, image=new_image)
