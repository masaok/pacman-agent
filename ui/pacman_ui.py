# Pacman's "Face" UI

# Credit: https://stackoverflow.com/questions/8176599/drawing-pacmans-face-in-tkinter

# TODO: Make the face toggle open and close on each move?

from tkinter import *

from constants import Constants


class PacmanUI:
    def __init__(self, canvas):
        self.canvas = canvas
        self.start_angle = 45   # zero is east
        self.stop_angle = 270   # relative to start angle counter clockwise

        self.start_dead_angle = 315  # zero is east
        self.stop_dead_angle = 270   # relative to start angle counter clockwise

        self.x = 100
        self.y = 240
        self.rad = 35  # radius

    # Creates an arc
    def draw(self, token, x, y):
        if token == Constants.PLR_BLOCK:
            color = 'yellow'
            self.canvas.create_arc(
                x - self.rad, y - self.rad, x + self.rad, y + self.rad, fill=color, style=PIESLICE,
                start=self.start_angle, extent=self.stop_angle)
        else:
            color = 'red'
            self.canvas.create_arc(
                x - self.rad, y - self.rad, x + self.rad, y + self.rad, fill=color, style=PIESLICE,
                start=self.start_dead_angle, extent=self.stop_dead_angle)
