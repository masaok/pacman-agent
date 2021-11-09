import tkinter as tk

from pprint import pprint


class MazeUI:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, window, maze):
        print("MAZE UI INIT")
        self._maze = maze
        # self._window = tk.Tk()
        # self._window.geometry("500x500")

        self._rows = len(maze)
        self._cols = len(maze[0])
        print("ROWS: ", self._rows)
        print("COLS: ", self._cols)

        self._block_size = 80

        canvas_width = self._cols * self._block_size
        canvas_height = self._rows * self._block_size

        w = canvas_width
        h = canvas_height

        # get screen width and height
        ws = window.winfo_screenwidth()  # width of the screen
        hs = window.winfo_screenheight()  # height of the screen

        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        # set the dimensions of the screen
        # and where it is placed
        # window.geometry('%dx%d+%d+%d' % (w, h, x, y))
        window.geometry('+%d+%d' % (x, y))
        self._window = window

        self._canvas = tk.Canvas(
            self._window, height=canvas_height, width=canvas_width)
        self._canvas.grid(row=0, column=0, sticky='w')

    ##################################################################
    # Methods
    ##################################################################

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
                else:
                    self._canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                                  tags="area")

        self._window.update()
