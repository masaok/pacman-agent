# https://stackoverflow.com/questions/47849592/moving-circle-using-tkinter

import tkinter as tk
import random


def move():
    global x
    global y
    global x_vel
    global y_vel

    # get random move
    x_vel = random.randint(-5, 5)
    y_vel = -5

    canvas1.move(circle, x_vel, y_vel)
    coordinates = canvas1.coords(circle)

    x = coordinates[0]
    y = coordinates[1]

    # if outside screen move to start position
    if y < -height:
        x = start_x
        y = start_y
        canvas1.coords(circle, x, y, x+width, y+height)

    window.after(33, move)

# --- main ---


start_x = 100
start_y = 100

x = start_x
y = start_y

width = 50
height = 50

x_vel = 5
y_vel = 5

window = tk.Tk()
window.geometry("1000x1000")

canvas1 = tk.Canvas(window, height=1000, width=1000)
canvas1.grid(row=0, column=0, sticky='w')

coord = [x, y, x+width, y+height]
circle = canvas1.create_oval(coord, outline="black", fill="red")

x = 200
y = 200
coord = [x, y, x+width, y+height]
circle = canvas1.create_oval(coord, outline="black", fill="green")

x = 100
y = 200
coord = [x, y, x+width, y+height]
circle = canvas1.create_oval(coord, outline="black", fill="purple")

x = 200
y = 100
coord = [x, y, x+width, y+height]
circle = canvas1.create_oval(coord, outline="black", fill="pink")

# move()

window.mainloop()
