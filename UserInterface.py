from functools import partial
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2

import FeatureExtraction
import SVMModel

root = Tk()


def select_image():
    img_path = filedialog.askopenfilename()

    image = Image.open(img_path)
    image = image.resize((800, 600), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)

    panel = Label(root, image=image)
    panel.image = image
    panel.grid(row=2)

def load_models():
    print("loading models")

if __name__ == "__main__":
    # Set Title as Image Loader
    root.title("Image Loader")

    # Set the resolution of window
    root.geometry('800x600')

    # Allow Window to be resizable
    root.resizable(width=True, height=True)

    # Create a button and place it into the window using grid layout
    btn = Button(root, text='load image', command=select_image).grid(
        row=1, column=0, columnspan=4)
    btn
    btn_load_models = Button(root, text='Load Trained Models', command=load_models).grid(
        row=1, column=6, columnspan=4)
    root.mainloop()
