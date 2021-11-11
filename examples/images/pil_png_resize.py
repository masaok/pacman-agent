# https://stackoverflow.com/questions/56554692/unable-to-put-transparent-png-over-a-normal-image-python-tkinter/56555164#56555164
# https://www.tutorialspoint.com/how-to-resize-an-image-using-tkinter

from tkinter import *
from PIL import Image, ImageTk


root = Tk()
root.title("Game")


frame = Frame(root)
frame.pack()


canvas = Canvas(frame, bg="black", width=700, height=400)
canvas.pack()

# Create Image on Canvas
# https://stackoverflow.com/questions/65290953/how-do-i-use-the-create-image-in-tkinter-in-python

background = PhotoImage(file="background.png")
canvas.create_image(350, 200, image=background)

desired_height = 80
desired_width = 80

character = Image.open("red_ghost_trans.png")
print(character)

character = character.resize((desired_width,desired_height), Image.ANTIALIAS)

character = ImageTk.PhotoImage(character)

canvas.create_image(100,200, anchor=NW, image=character)


root.mainloop()
