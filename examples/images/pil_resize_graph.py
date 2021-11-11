# https://www.tutorialspoint.com/how-to-resize-an-image-using-tkinter

# Import the required Libraries
from tkinter import *
from PIL import Image, ImageTk

# Create an instance of tkinter frame
win = Tk()

# Set the geometry of tkinter frame
# win.geometry("600x600")

# Create a canvas
canvas = Canvas(win, width=500, height=400)

# What does pack do?
# https://riptutorial.com/tkinter/example/29712/pack--
canvas.pack()

# Load an image in the script
img = (Image.open("../images/red_ghost_trans.png"))
print("img: ", img)

# Resize the Image using resize method
img = img.resize((200, 200), Image.ANTIALIAS)
print("resized: ", img)
img = ImageTk.PhotoImage(img)
print("new_image: ", img)

# Add image to the Canvas Items
canvas.create_image(0, 0, anchor=NW, image=img)

win.mainloop()
