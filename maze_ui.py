'''
Graphical component for rendering the Pacman mazes.
'''

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from pprint import pprint
from constants import Constants
from ui.pacman_ui import PacmanUI
from ui.ghost_ui import GhostUI


class MazeUI:

    def __init__(self, window, maze, debug=False):

        # Save the window to add buttons later
        self.window = window
        self._debug = debug

        if self._debug:
            print("MAZE UI INIT")
        self._maze = maze

        self._rows = len(maze)
        self._cols = len(maze[0])
        if self._debug:
            print("ROWS: ", self._rows)
            print("COLS: ", self._cols)

        self._block_size = 80

        self.canvas_width = self._cols * self._block_size
        self.canvas_height = self._rows * self._block_size

        self.controls_height = 100

        # Leave room for buttons at the bottom
        self.canvas_height += self.controls_height

        w = self.canvas_width
        h = self.canvas_height

        # get screen width and height
        ws = window.winfo_screenwidth()  # width of the screen
        hs = window.winfo_screenheight()  # height of the screen

        # Calculate x and y coordinates for the *centered* Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        
        if self._debug:
            print("MAZE UI x: ", x)
            print("MAZE UI y: ", y)

        # set the dimensions of the screen
        # and where it is placed
        window.geometry('+%d+%d' % (x, y))
        self.window = window

        self._canvas = tk.Canvas(
            self.window, height=self.canvas_height, width=self.canvas_width)
        # self._canvas.grid(row=0, column=0, sticky='w')
        self._canvas.pack()


        # Draw controls
        self.btn_var = tk.IntVar()
        self.btn = Button(self.window, text='Step forward', width=40,
                          height=5, bd='10', command=lambda: self.btn_var.set(1))
        self.exit_button = Button(self.window, text='Exit', width=40,
                                  height=5, bd='10', command=self.destroy)

        # Initialize Pacman UI
        self.pacman = PacmanUI(self._canvas)
        self.ghost = GhostUI(self._canvas)

        self.window.update()

    def destroy(self):
        self.window.destroy()

    def draw_maze(self):
        if self._debug:
            pprint(self._maze)

        self._canvas.delete('all')

        for row in range(self._rows):
            for col in range(self._cols):

                # Get the symbol
                token = self._maze[row][col]
                color = "black"
                if token == Constants.PLR_BLOCK:
                    color = "yellow"
                elif token == Constants.SAFE_BLOCK:
                    color = "black"
                elif token == Constants.GHOST_BLOCK:
                    color = "pink"
                elif token == Constants.PELLET_BLOCK:
                    color = "white"
                elif token == Constants.DEATH_BLOCK:
                    color = "red"
                elif token == Constants.WALL_BLOCK:
                    color = "blue"

                x1 = (col * self._block_size)
                y1 = row * self._block_size
                x2 = x1 + self._block_size
                y2 = y1 + self._block_size

                if token == Constants.PELLET_BLOCK:
                    margin = 25
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                    self._canvas.create_oval(
                        [x1 + margin, y1 + margin, x2 - margin, y2 - margin],
                        fill=color)
                elif token == Constants.PLR_BLOCK or token == Constants.DEATH_BLOCK:
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                    self.pacman.draw(token, (x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
                elif token == Constants.GHOST_BLOCK:
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                    self.ghost.draw(x1, y1, self._block_size, self._block_size)
                else:
                    self._canvas.create_rectangle(
                        x1, y1, x2, y2, fill=color, outline="black", width=3, tags="area")

        # Draw controls
        self._canvas.create_rectangle(
            0, self.canvas_height - self.controls_height, self.canvas_width, self.canvas_height,
            fill="black")

        self.btn.place(x=100, y=(self.canvas_height - self.controls_height))

        # Force the update so the window shows immediately
        # https://python-forum.io/thread-34564.html
        self.window.update()
