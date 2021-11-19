# Ghost UI

from tkinter import *
from PIL import Image, ImageTk


class GhostUI:
    def __init__(self, canvas):
        self.canvas = canvas
        self.filename = "ui/images/red_ghost_trans.png"
        self.image = Image.open(self.filename)

        self.start_angle = 0   # zero is east
        self.stop_angle = 180   # relative to start angle counter clockwise

        self.rad = 25  # radius

        self.color = 'pink'

        self.cx = 0  # center of the block
        self.cy = 0  # center of the block

        self.upper_body_height = 10  # above the center
        self.lower_body_height = 11  # below the center

        self.eye_radius = 6
        self.eye_height_offset = 5
        self.eye_distance_offset = 2
        self.eye_border_width = 0

        self.eyeball_color = "white"

        self.pupil_radius = 4
        self.pupil_offset = 2

        self.left_eye_x = 0
        self.left_eye_y = 0

        self.right_eye_x = 0
        self.right_eye_y = 0

    def draw_circle(self, x, y, w, h):
        margin = 10
        self.canvas.create_oval(
            [x + margin, y + margin, x + w - margin, y + h - margin],
            fill="pink")

        # Create Text: https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/create_text.html
        # Anchors: https://www.tutorialspoint.com/python/tk_anchors.htm
        font = "Arial 30 bold"
        self.canvas.create_text(x + 0.5 * w, y + 0.5 * h, anchor=CENTER,
                                font=font, text="G", fill="red")

    def draw_eyeball(self, ecx, ecy, color="white"):
        x1 = ecx - self.eye_radius
        y1 = ecy - self.eye_radius

        x2 = ecx + self.eye_radius
        y2 = ecy + self.eye_radius

        points = [x1, y1, x2, y2]

        self.canvas.create_oval(points, fill=color, width=self.eye_border_width)

    def draw_pupil(self, ecx, ecy, color="black"):
        x1 = ecx - self.pupil_radius
        y1 = ecy - self.pupil_radius

        x2 = ecx + self.pupil_radius
        y2 = ecy + self.pupil_radius

        points = [x1, y1, x2, y2]

        self.canvas.create_oval(points, fill=color, width=self.eye_border_width)

    def set_center(self, x, y, w, h):
        self.cx = (w / 2) + x  # center within the block
        self.cy = (h / 2) + y  # center within the block

    def draw_head(self, x, y, w, h):
        x = self.cx
        y = self.cy - 10
        self.canvas.create_arc(
            x - self.rad, y - self.rad, x + self.rad, y + self.rad, fill=self.color, style=PIESLICE,
            start=self.start_angle, extent=self.stop_angle)

    def draw_shapes(self, x, y, w, h):
        color = self.color

        self.set_center(x, y, w, h)
        cx = self.cx
        cy = self.cy

        # Draw the head
        self.draw_head(x, y, w, h)

        # Draw the body
        x1 = cx - self.rad + 1  # small adjustment to line up the body with the head (on Mac)
        x2 = cx + self.rad

        y1 = cy - self.upper_body_height
        y2 = cy + self.lower_body_height

        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill=color, width=0)

        # Draw the left foot
        x1 = cx - self.rad
        y1 = cy + 10

        x2 = cx - self.rad + (2/3 * self.rad)
        y2 = cy + 10

        x3 = cx - self.rad + (1/3 * self.rad)
        y3 = cy + 10 + (1/3 * self.rad)

        points = [x1, y1, x2, y2, x3, y3]  # order does not matter
        self.canvas.create_polygon(points, fill=color)

        # Draw the middle foot
        x1 = cx - self.rad + (2/3 * self.rad)
        y1 = cy + 10

        x2 = x1 + (2/3 * self.rad)
        y2 = cy + 10

        x3 = cx
        y3 = cy + 10 + (1/3 * self.rad)

        points = [x1, y1, x2, y2, x3, y3]  # order does not matter
        self.canvas.create_polygon(points, fill=color)

        # Draw the right foot
        x1 = cx + (1/3 * self.rad)
        y1 = cy + 10

        x2 = cx + self.rad
        y2 = cy + 10

        x3 = cx + (2/3 * self.rad)
        y3 = cy + 10 + (1/3 * self.rad)

        points = [x1, y1, x2, y2, x3, y3]  # order does not matter
        self.canvas.create_polygon(points, fill=color)

        # Draw the left eye
        self.left_eye_x = self.cx - self.rad + (2/3 * self.rad) - self.eye_distance_offset
        self.left_eye_y = self.cy - self.upper_body_height - self.eye_height_offset
        self.draw_eyeball(self.left_eye_x, self.left_eye_y, color=self.eyeball_color)

        # Draw the right eye
        self.right_eye_x = self.cx + (1/3 * self.rad) + self.eye_distance_offset
        self.right_eye_y = self.cy - self.upper_body_height - self.eye_height_offset
        self.draw_eyeball(self.right_eye_x, self.right_eye_y, color=self.eyeball_color)

        # Draw left pupil
        ecx = self.left_eye_x + self.pupil_offset
        ecy = self.left_eye_y
        self.draw_pupil(ecx, ecy)

        # Draw right pupil
        ecx = self.right_eye_x + self.pupil_offset
        ecy = self.right_eye_y
        self.draw_pupil(ecx, ecy)

    def draw(self, x, y, w, h):
        # self.draw_circle(x, y, w, h)  # the original G circle
        self.draw_shapes(x, y, w, h)

        # TODO: None of this image stuff works yet
        # self.canvas.pack()

        # print("x: ", x)
        # print("y: ", y)
        # print("w: ", w)
        # print("h: ", h)
        # print(self.image)

        # self.image = self.image.resize((w, h), Image.ANTIALIAS)
        # new_image = ImageTk.PhotoImage(self.image)
        # print(new_image)
        # self.canvas.create_image(x, y, anchor=NW, image=new_image)
        # self.canvas.create_image(x, y, image=new_image)
        # self.canvas.create_image(0, 0, anchor=NW, image=new_image)
