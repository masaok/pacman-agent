# Credit: https://stackoverflow.com/questions/8176599/drawing-pacmans-face-in-tkinter

from tkinter import *


class Application(Frame):

    # Creates an arc
    def draw_pieslice(self, canv, x, y, rad):
        return canv.create_arc(
            x - rad, y - rad, x + rad, y + rad, fill='yellow', style=PIESLICE, start=self.start_angle,
            extent=self.stop_angle)

    # To toggle the start and extent values of the arc, such that the pacman opens and closes the mouth
    # Initially the mouth will be completely closed, but as the value is toggled we will get a 45deg space (mouth opens)
    def toggle_mouth(self):
        if self.start_angle == 1:
            self.start_angle = 45
            self.stop_angle = 270
        else:
            self.start_angle = 1
            self.stop_angle = 359

    # moves the pacman body horizontally
    def movecircle(self):
        self.repeat = self.repeat - 1  # sets a limit to horizontal movement
        self.canvas.move(self.PacMouth, 1, 0)
        if (self.repeat % 10) == 0:  # Adjusting the value in here, will help to adjust the mouth movement speed with the horizontal body movement
            self.toggle_mouth()
            self.canvas.itemconfig(
                self.PacMouth, start=self.start_angle, extent=self.stop_angle)
        if self.repeat != 0:
            # By adjusting the time, we can adjust the horizontal movement speed
            self.after(10, self.movecircle)

    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.start_angle = 1
        self.stop_angle = 359

        self.canvas = Canvas(width=800, height=480, bg='black')
        self.canvas.pack(expand=YES, fill=BOTH)
        text = self.canvas.create_text(
            50, 10, text="Pacman! Yeah!", fill="white")
        self.PacMouth = self.draw_pieslice(self.canvas, 100, 240, 20)
        self.repeat = 600

        self.movecircle()


root = Tk()
root.config(bg="black")
root.title("Im on stack overflow")
root.geometry('{}x{}'.format(800, 480))
root.resizable(width=False, height=False)

app = Application(master=root)
app.mainloop()
