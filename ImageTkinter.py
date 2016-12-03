import os
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk


if __name__ == "__main__":
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)


    list_of_images = []

    #adding the image
    # File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    # print(File)
    for root, dirs, files in os.walk("/Users/Sri/Desktop/picture", topdown=False):
        for name in files:
            current_image_path = os.path.join(root, name)
            list_of_images.append(current_image_path)


    img = ImageTk.PhotoImage(Image.open(list_of_images[0]))

    current_canvas_image_id = canvas.create_image(0,0,image=img,anchor="nw")


    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        print (event.x,event.y)

    # def bind_key_event(event):
    #     if event.keysym == 'Right':
    #         canvas.delete(current_canvas_image_id)
    #         current_canvas_image_id = canvas.create_image(0, 0, image=img, anchor="nw")
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)
    # canvas.bind_all("<KeyPress-Right>", bind_key_event)

    root.mainloop()