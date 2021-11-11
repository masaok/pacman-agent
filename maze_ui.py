import tkinter as tk
from tkinter import *

from pprint import pprint

from ui.pacman_ui import PacmanUI


class MazeUI:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, window, maze):

        # Save the window to add buttons later
        self.window = window

        print("MAZE UI INIT")
        self._maze = maze
        # self._window = tk.Tk()
        # self._window.geometry("500x500")

        self._rows = len(maze)
        self._cols = len(maze[0])
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

        # set the dimensions of the screen
        # and where it is placed
        # window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        window.geometry('+%d+%d' % (x, y))
        self._window = window

        self._canvas = tk.Canvas(
            self._window, height=self.canvas_height, width=self.canvas_width)
        self._canvas.grid(row=0, column=0, sticky='w')

        # Draw controls
        # TODO: Don't draw the controls every render
        self.btn_var = tk.IntVar()
        self.btn = Button(self.window, text='Step forward', width=40,
                          height=5, bd='10', command=lambda: self.btn_var.set(1))
        self.exit_button = Button(self.window, text='Exit', width=40,
                                  height=5, bd='10', command=self.destroy)

        # Initialize Pacman UI
        self.pacman = PacmanUI(self._canvas)

    ##################################################################
    # Methods
    ##################################################################

    def destroy(self):
        self.window.destroy()

    def draw_maze(self):
        pprint(self._maze)

        self._canvas.delete('all')

        for row in range(self._rows):
            for col in range(self._cols):

                # Get the symbol
                token = self._maze[row][col]
                # print("ROW: ", row, " COL: ", col, " TOKEN: ", token)

                color = "black"
                if token == '@':
                    color = "yellow"
                elif token == '.':
                    color = "black"
                elif token == 'G':
                    color = "pink"
                elif token == 'P':
                    color = "white"
                elif token == 'E':
                    color = "green"
                elif token == 'X':
                    color = "blue"

                x1 = (col * self._block_size)
                # y1 = ((7 - row) * self.dim_square)
                y1 = row * self._block_size
                x2 = x1 + self._block_size
                y2 = y1 + self._block_size

                if token == 'P':
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                    self._canvas.create_oval([x1, y1, x2, y2], fill=color)
                elif token == '@':
                    # self._canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                    #                               tags="area")
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                    self.pacman.draw_pieslice((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
                else:
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                                  tags="area")

        # Draw controls
        # TODO: Don't draw the controls every render
        self._canvas.create_rectangle(
            0, self.canvas_height - self.controls_height, self.canvas_width, self.canvas_height,
            fill="black")

        self.btn.place(x=100, y=(self.canvas_height - self.controls_height))
        # self.exit_button.place(
        #     x=300, y=(self.canvas_height - self.controls_height))

        # Force the update so the window shows immediately
        # https://python-forum.io/thread-34564.html
        self._window.update()
