# https://www.tutorialspoint.com/what-does-the-wait-window-method-do-in-tkinter

# Import the required libraries
from tkinter import *

# Create an instance of tkinter frame
win = Tk()

# Set the size of the tkinter window
win.geometry("700x350")

# Add a Text widget in a toplevel window
top = Toplevel(win)
top.geometry("450x150")
Label(top, text="This is a TopLevel Window", font=('Aerial 17')).pack(pady=50)

# Wait for the toplevel window to be closed
win.wait_window(top)
print("Top Level Window has been Closed!")
win.destroy()

win.mainloop()
