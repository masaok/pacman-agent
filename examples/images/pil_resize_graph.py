# https://www.tutorialspoint.com/how-to-resize-an-image-using-tkinter

#Import the required Libraries
from tkinter import *
from PIL import Image,ImageTk

#Create an instance of tkinter frame
win = Tk()

#Set the geometry of tkinter frame
win.geometry("750x270")

#Create a canvas
canvas= Canvas(win, width= 600, height= 400)

# What does pack do?
# https://riptutorial.com/tkinter/example/29712/pack--
canvas.pack()

#Load an image in the script
img= (Image.open("hero.png"))

#Resize the Image using resize method
resized_image= img.resize((300,205), Image.ANTIALIAS)
new_image= ImageTk.PhotoImage(resized_image)

#Add image to the Canvas Items
canvas.create_image(10,10, anchor=NW, image=new_image)

win.mainloop()