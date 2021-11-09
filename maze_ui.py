import tkinter as tk

from pprint import pprint


class MazeUI:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__(self, maze):

        print("MAZE UI maze:")
        pprint(maze)

        self.env = maze

        start_x = 230
        start_y = 270

        x = start_x
        y = start_y

        width = 50
        height = 50

        self._window = tk.Tk()
        self._window.geometry("1000x1000")

        self._canvas1 = tk.Canvas(self._window, height=1000, width=1000)
        self._canvas1.grid(row=0, column=0, sticky='w')

        coord = [x, y, x+width, y+height]
        self._circle = self._canvas1.create_oval(
            coord, outline="red", fill="red")

        coord = [x, y, x+40, y+40]
        self._rect2 = self._canvas1.create_rectangle(
            coord, outline="Blue", fill="Blue")

        self._window.mainloop()

    ##################################################################
    # Methods
    ##################################################################
