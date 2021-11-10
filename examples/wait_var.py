# https://stackoverflow.com/questions/44790449/making-tkinter-wait-untill-button-is-pressed

import tkinter as tk

# Open a window
# Show the button
# Wait until the button is pressed

root = tk.Tk()
var = tk.IntVar()
button = tk.Button(root, text="Click Me", command=lambda: var.set(1))
button.place(relx=.5, rely=.5, anchor="c")

print("waiting...")
button.wait_variable(var)
print("done waiting.")
