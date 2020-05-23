import concurrent.futures
import csv
import glob
from concurrent.futures import wait, ALL_COMPLETED

import cv2
import numpy as np

filename = "D:/leafimages/scanning/n21-12-123.jpg"
filename1 = "D:\leafimages\procImages\LSheath0.png"

leaf_features = []
sheath_features = []


def main():
    img_dir = glob.glob("D:/leafimages/scanning/*.jpg")

    # count = 1
    # for image in img_dir:
    #     class_name = image.rsplit("\\", 1)[1]
    #     class_name = class_name[0]
    #     class_name = class_type(class_name)
    #     input_img = cv2.imread(image)
    #     print("processing: ", count)
    #     separate_leaf_and_sheath(input_img, class_name, image)
    #     count += 1
    #
    # print("leaf features", np.asarray(leaf_features).shape)
    #
    # print("sheath features", np.asarray(sheath_features).shape)

    # input_img = cv2.imread(filename)
    # cv2.waitKey(0)
    # separate_leaf_and_sheath(input_img)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_features = {
            executor.submit(separate_leaf_and_sheath, cv2.imread(image), class_type(image.rsplit("\\", 1)[1][0]),
                            image): image for image in img_dir}
        wait(future_to_features, timeout=None, return_when=ALL_COMPLETED)
        # for future in concurrent.futures.as_completed(future_to_features):
        #     img_name = future_to_features[future]
        #     try:
        #         ret_str = future.result()
        #     except Exception as ex:
        #         print("Exception Occured in:", img_name)
        #     else:
        #         print(ret_str)

    print("leaf features", np.asarray(leaf_features).shape)
    print("sheath features", np.asarray(sheath_features).shape)

    with (open('D:\leafimages\leaf_features.csv', 'w', newline='')) as file:
        writer = csv.writer(file, dialect='excel')
        for l_feature in leaf_features:
            writer.writerow(l_feature)

    with (open('D:\leafimages\sheath_features.csv', 'w', newline='')) as file:
        writer = csv.writer(file, dialect='excel')
        for s_feature in sheath_features:
            writer.writerow(s_feature)


def separate_leaf_and_sheath(input_image, plant_class=None, img_name=None):
    print("Started", img_name)

    # hsv color space
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsvimage', cv2.resize(hsv_image, (800, 600)))  # debug
    # cv2.imwrite("D:\leafimages\procImages\HSVImage.jpg", hsv_image)  # debug

    # leafcolorvalues1 = np.array([10, 25, 25])  # upper bound
    # leafcolorvalues2 = np.array([70, 255, 255])  # lower bound
    leafcolorvalues1 = np.array([0, 25, 25])  # upper bound
    leafcolorvalues2 = np.array([70, 255, 255])  # lower bound

    mask_image = cv2.inRange(hsv_image, leafcolorvalues1, leafcolorvalues2)

    # cv2.imshow('maskimage', cv2.resize(mask_image, (800, 600)))  # debug
    # cv2.imwrite("D:\leafimages\procImages\Hmaskimage.jpg", mask_image)  # debug

    # finding the contours
    contours = cv2.findContours(mask_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    img_num = 0

    # calculating average of contour width and area
    total_area = 0
    total_width = 0
    total_green = 0
    total_arc_len = 0
    num_items = 0
    for cntr in contours:
        contour_area = cv2.contourArea(cntr)
        if contour_area < 45000:
            continue
        rgb_mean = rgb_mean_of_contour(input_image, cntr)
        cropped_rotated_img = cropped_rotated(input_image, cntr)
        height, width, channel = cropped_rotated_img.shape
        num_items += 1
        total_area += contour_area
        total_width += width
        total_green += rgb_mean[1]
        total_arc_len += cv2.arcLength(cntr, 0)
    average_width = total_width / num_items
    average_area = total_area / num_items
    average_green = total_green / num_items
    average_arclen = total_arc_len / num_items
    # end of calculating averages

    # iterating through each contour
    for cntr in contours:
        contour_area = cv2.contourArea(cntr)
        if contour_area < 45000:
            continue

        cropped_rotated_img = cropped_rotated(input_image, cntr)

        # classifying leaf or sheath based on average area
        if cv2.contourArea(cntr) > average_area:
            # cv2.imwrite("D:\leafimages\procImages\Leaf_{}.jpg".format(img_num), cropped_rotated_img)  # debug
            # arr_leaf.append(rgb_mean)
            # print(str(img_num) + " height: " + str(cropped_rotated_img.shape[0]) + " width: " + str(
            #     cropped_rotated_img.shape[1]))  # debug

            l_features = []  # single leaf features

            # feature 1 Leaf R, G, B
            leaf_bgr_val = rgb_mean_of_contour(input_image, cntr)
            l_features.append(leaf_bgr_val[2])
            l_features.append(leaf_bgr_val[1])
            l_features.append(leaf_bgr_val[0])

            # feature 2 LEAF TIP R, G, B
            height, width, channels = cropped_rotated_img.shape
            w_start = int(width * 0.8)
            leaf_tip = cropped_rotated_img[0:height, w_start: width]
            bgr_val = rgb_leaf_tip(leaf_tip)
            l_features.append(bgr_val[2])
            l_features.append(bgr_val[1])
            l_features.append(bgr_val[0])

            # feature 3 LEAF AREA
            l_features.append(contour_area)

            # feature 4 LEAF LENGTH
            l_features.append(width)

            # feature 5 LEAF AREA LENGTH RATIO
            l_features.append(contour_area / width)

            # adding the plant_class
            l_features.append(plant_class)

            # print(l_features)  # debug

            leaf_features.append(l_features)

        else:
            # cv2.imwrite("D:\leafimages\procImages\LSheath{}.jpg".format(img_num), cropped_rotated_img)  # debug
            # arr_sheath.append(rgb_mean)
            # print(str(img_num) + "height: " + str(cropped_rotated_img.shape[0]) + "width: " + str(
            #     cropped_rotated_img.shape[1]))

            s_features = []  # single sheath features

            # feature 1
            sheath_bgr_val = rgb_mean_of_contour(input_image, cntr)
            s_features.append(sheath_bgr_val[2])
            s_features.append(sheath_bgr_val[1])
            s_features.append(sheath_bgr_val[0])
            s_features.append(plant_class)

            # print(s_features)  # debug

            sheath_features.append(s_features)

        img_num += 1
        # arr_toRet.append(cropped)
        # str_to_ret = "completed extracting feature from: " + img_name
        print("Completed", img_name)  # debug
        # return str_to_ret
    # cv2.imshow('drawContours', cv2.resize(input_image, (800, 600)))  # debug
    # cv2.waitKey(0)
    # return arr_sheath, arr_leaf


# return cropped rotated image
def cropped_rotated(original_image, contour):
    multiplier = 1

    rectangle = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)
    # cv2.drawContours(input_image, [box], 0, (0, 191, 255), 5)  # debug

    # to get the width and the height of the rectangle
    W = rectangle[1][0]
    H = rectangle[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rectangle[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(multiplier * (x2 - x1)), int(multiplier * (y2 - y1)))
    # cv2.circle(inputImage, center, 10, (0, 255, 0), -1)  # debug

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(original_image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    cropped_width = W if not rotated else H
    cropped_height = H if not rotated else W

    croppedRotated = cv2.getRectSubPix(cropped, (int(cropped_width * multiplier), int(cropped_height * multiplier)),
                                       (size[0] / 2, size[1] / 2))

    return croppedRotated


# given contour and the original image calculates the mean RGB value of the contour
def rgb_mean_of_contour(input_image, contour):
    mask = np.zeros(input_image.shape[:2], np.uint8)
    cv2.drawContours(mask, contour, -1, 255, -1)
    # cv2.imshow("contours", cv2.resize(mask, (800, 600)))  # debug
    # cv2.waitKey(0)

    rgb_mean = cv2.mean(input_image, mask)

    return rgb_mean


def rgb_leaf_tip(img):
    # hsv_leaf = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_leaf = cv2.split(hsv_leaf)
    # gray_leaf = hsv_leaf[0]
    #
    # gray_leaf = cv2.bitwise_not(gray_leaf)
    #
    # gray_leaf = cv2.GaussianBlur(gray_leaf, (5, 5), sigmaX=-1, sigmaY=-1)
    #
    # binary_img = cv2.threshold(gray_leaf, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    leafcolorvalues1 = np.array([0, 25, 25])  # upper bound
    leafcolorvalues2 = np.array([70, 255, 255])  # lower bound

    mask_image = cv2.inRange(hsv_image, leafcolorvalues1, leafcolorvalues2)

    # cv2.imshow("binary_img", mask_image)  # debug

    img_contours = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    rgb_mean = rgb_mean_of_contour(img, img_contours)

    return rgb_mean

    #    print("img_contours", img_contours.shape)

    # cv2.drawContours(img, img_contours, -1, 255, 2)
    # cv2.imshow("contours", img)  # debug

    # mask = np.zeros(img.shape[:2], np.uint8)
    # print(mask.shape)  # debug
    # cv2.drawContours(mask, img_contours, -1, 255, -1)  # debug
    # cv2.imshow("maskImage", mask)  # debug

    # cv2.imshow("img", img)  # debug
    # cv2.waitKey(0)


def class_type(char):
    if char == "k":
        return "POTASSIUM"

    if char == "n":
        return "NITROGEN"

    if char == "p":
        return "PHOSPHORUS"

    if char == "o":
        return "NORMAL"

    return ""


if __name__ == "__main__":
    main()

# references:
# https://stackoverflow.com/questions/36921249/drawing-angled-rectangles-in-opencv
# https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
# https://stackoverflow.com/questions/54316588/get-the-average-color-inside-a-contour-with-open-cv
# https://stackoverflow.com/questions/54316588/get-the-average-color-inside-a-contour-with-open-cv
