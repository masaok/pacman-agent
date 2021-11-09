# https://www.tutorialspoint.com/how-to-clear-tkinter-canvas

import time

# Import the tkinter library
from tkinter import *


def move():
    global index

    print("index: ", index)

    if (index >= 1):  # run after 1 second
        canvas1.delete('all')
    
    if (index >= 2):
        window.destroy()

    index += 1

    window.after(1000, move)


index = 0

start_x = 230
start_y = 270

x = start_x
y = start_y

width = 50
height = 50

# Create an instance of tkinter frame
window = Tk()

# Set the geometry
window.geometry("650x250")

# Creating a canvas
canvas1 = Canvas(window, bg="white", height=200, width=200)
cordinates = 10, 10, 200, 200
arc = canvas1.create_arc(cordinates, start=0, extent=320, fill="red")
canvas1.pack()

coord = [x, y, x+width, y+height]
circle = canvas1.create_oval(coord, outline="red", fill="red")

# time.sleep(1)

# Clearing the canvas

move()

window.mainloop()
