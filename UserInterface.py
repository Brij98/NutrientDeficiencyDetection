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
    image = image.resize((250, 250), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)

    panel = Label(root, image=image)
    panel.image = image
    panel.grid(column=2)

    # l_features, s_features = FeatureExtraction.separate_leaf_and_sheath(cv2.imread(img_path), predict=True)
    # print(l_features)
    # # new_f = []
    # # for l in l_features:
    # #     temp = []
    # #     temp.append(l[1])
    # #     temp.append(l[3])
    # #     temp.append(l[4])
    # #     temp.append(l[6])
    # #     new_f.append(temp)
    l_features, s_features = FeatureExtraction.predict_normal_features(cv2.imread(img_path))
    result = SVMModel.predict_normal_leaf_model(l_features)
    print(result)


def predict_class():
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

    btn_load_models = Button(root, text='Predict', command=predict_class).grid(
        row=3, column=0, columnspan=4)
    root.mainloop()


