from tkinter import Tk, Text, BOTH, W, N, E, S, filedialog
from tkinter.ttk import Frame, Button, Label, Style

import cv2
from PIL import Image
from PIL import ImageTk

import tkinter as tk

import FeatureExtraction
import LogisticRegressionModel
import SVMModel


class SystemGUI(Frame):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.master.title("Image Loader")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        lbl = Label(self, text="Image")
        lbl.grid(sticky=W, pady=4, padx=5)

        panel = Label(self)
        panel.grid(row=1, column=0, columnspan=2, rowspan=4,
                   padx=5, sticky=E + W + S + N)

        # area = Text(self)
        # area.grid(row=1, column=0, columnspan=2, rowspan=4,
        #           padx=5, sticky=E + W + S + N)

        abtn = Button(self, text="Load Image", width=30, command=self.select_image)
        abtn.grid(row=1, column=3)

        cbtn = Button(self, text="NPK Classification", width=30, command=self.classify_plant)
        cbtn.grid(row=2, column=3)

        # hbtn = Button(self, text="Classify")
        # hbtn.grid(row=5, column=0, padx=5)
        global area
        area = Text(self, width=45)
        area.grid(row=3, column=3, columnspan=5, rowspan=2,
                  padx=5, pady=5, sticky=S + N)

        # gbtn = Button(self, text="Clear", width=30, command=self.classify_plant)
        # gbtn.grid(row=7, column=3)

        # obtn = Button(self, text="OK")
        # obtn.grid(row=5, column=3)
        # area = Text(self)
        # area.grid(row=5, column=0)

    def select_image(self):
        global img_path
        img_path = filedialog.askopenfilename()

        image = Image.open(img_path)
        image = image.resize((600, 600), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)

        panel = Label(self, image=image)
        panel.image = image
        panel.grid(row=1, column=0, columnspan=2, rowspan=4,
                   padx=5, sticky=E + W + S + N)

    def classify_plant(self):
        # img = cv2.imread(img_path)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        #  Normal not normal classification

        l_features, s_features = FeatureExtraction.predict_normal_features(cv2.imread(img_path))
        l_results = SVMModel.predict_normal_leaf_model(l_features)
        s_results = LogisticRegressionModel.predict_normal_sheath_model(s_features)

        output_str = " Normal and Not Normal Classification \n"

        l_tot = 0
        s_tot = 0
        for l in range(len(l_results)):
            output_str += "Leaf " + str(l) + ": " + self.classification_label(0, l_results[l]) + "\n"
            l_tot += l_results[l]
        for s in range(len(s_results)):
            output_str += "sheath " + str(s) + ": " + self.classification_label(0, s_results[s]) + "\n"
            s_tot += s_results[s]

        output_str += "\n\n"

        #  NPK Classification
        if l_tot != 0 and s_tot != 0:

            l_features, s_features = FeatureExtraction.predict_npk_features(cv2.imread(img_path))
            l_results = SVMModel.predict_npk_leaf_model(l_features)
            s_results = LogisticRegressionModel.predict_npk_sheath_model(s_features)

            output_str += "NPK Classification \n"

            l_tot = 0
            s_tot = 0
            for l in range(len(l_results)):
                output_str += "Leaf " + str(l) + ": " + self.classification_label(1, l_results[l]) + "\n"
                l_tot += l_results[l]
            for s in range(len(s_results)):
                output_str += "sheath " + str(s) + ": " + self.classification_label(1, s_results[s]) + "\n"
                s_tot += s_results[s]

            output_str += "\n\n"

            # PK Classification
            if l_tot != 0 and s_tot != 0:

                l_features, s_features = FeatureExtraction.predict_pk_features(cv2.imread(img_path))
                l_results = SVMModel.predict_pk_leaf_model(l_features)
                s_results = LogisticRegressionModel.predict_pk_sheath_model(s_features)

                output_str += "PK Classification \n"

                for l in range(len(l_results)):
                    output_str += "Leaf " + str(l) + ": " + self.classification_label(2, l_results[l]) + "\n"
                for s in range(len(s_results)):
                    output_str += "sheath " + str(s) + ": " + self.classification_label(2, s_results[s]) + "\n"

            output_str += "\n\n"

        area.insert(tk.END, output_str)

    def classification_label(self, lvl, inclass):
        if lvl == 0:
            if inclass == 0:
                return "NORMAL"
            else:
                return "NPK"
        elif lvl == 1:
            if inclass == 0:
                return "NITROGEN"
            else:
                return "PK"
        else:
            if inclass == 0:
                return "PHOSPHORUS"
            else:
                return "POTASSIUM"


def main():
    root = Tk()
    root.geometry('1024x600')
    app = SystemGUI()
    root.mainloop()


if __name__ == '__main__':
    main()
