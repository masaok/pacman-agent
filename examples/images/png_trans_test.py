# https://stackoverflow.com/questions/3270209/how-do-i-make-tkinter-support-png-transparency
# from Tkinter import Tk, Frame, Canvas
from tkinter import *
import ImageTk

t = Tk()
t.title("Transparency")

frame = Frame(t)
frame.pack()

canvas = Canvas(frame, bg="black", width=500, height=500)
canvas.pack()

photoimage = ImageTk.PhotoImage(file="example.png")
canvas.create_image(150, 150, image=photoimage)

t.mainloop()
