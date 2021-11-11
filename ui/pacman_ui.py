
from tkinter import *


class PacmanUI:
    def __init__(self, canvas):
        self.canvas = canvas
        self.start_angle = 45
        self.stop_angle = 270

        self.x = 100
        self.y = 240
        self.rad = 30

    # Creates an arc
    def draw_pieslice(self, x, y):
        return self.canvas.create_arc(
            x - self.rad, y - self.rad, x + self.rad, y + self.rad, fill='yellow', style=PIESLICE,
            start=self.start_angle, extent=self.stop_angle)
