# https://stackoverflow.com/questions/56554692/unable-to-put-transparent-png-over-a-normal-image-python-tkinter/56555164#56555164

from tkinter import *

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

character = PhotoImage(file="hero.png")
h = character.height()
w = character.width()

print("height: ", h)
print("width: ", w)
# character = character.zoom(0.5, 0.5)
character = character.subsample(5, 5)

h = character.height()
w = character.width()

print("new height: ", h)
print("new width: ", w)


canvas.create_image(30, 30, image=character)

root.mainloop()
