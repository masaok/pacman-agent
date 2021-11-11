# https://www.tutorialspoint.com/how-to-resize-an-image-using-tkinter

# Import the required Libraries
from tkinter import *
from PIL import Image, ImageTk

# Create an instance of tkinter frame
win = Tk()

# Set the geometry of tkinter frame
win.geometry("750x600")

# Create a canvas
canvas = Canvas(win, width=600, height=600)

# What does pack do?
# https://riptutorial.com/tkinter/example/29712/pack--
canvas.pack()

# Load an image in the script
img = (Image.open("red_ghost_trans.png"))
print("img: ", img)

# Resize the Image using resize method
resized_image = img.resize((300, 300), Image.ANTIALIAS)
print("resized: ", resized_image)
new_image = ImageTk.PhotoImage(resized_image)
print("new_image: ", new_image)

# Add image to the Canvas Items
canvas.create_image(10, 10, anchor=NW, image=new_image)

win.mainloop()
