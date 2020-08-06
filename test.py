import os

import cv2
import numpy as np

import quickdraw


def crop(img):
    w, h = img.shape
    cx = 0
    cy = 0
    for j in range(h):
        for i in range(w):
            if img[j][i] != 0 and i > cx:
                cx = i
    for j in range(h):
        for i in range(w):
            if img[j][i] != 0 and j > cy:
                cy = j

    print(cy, cx)
    return img[:, :cx]


data_folder = '/home/mineorpe/work/restests/data/'
files = os.listdir(data_folder)
fp = 0
tp = 0
allp = len(files)
for file in files:
    imgc = quickdraw.classes[int(file.split("(")[0])]
    fname = data_folder + file
    digit2 = cv2.imread(fname)
    blackboard_gray = cv2.cvtColor(digit2, cv2.COLOR_BGR2GRAY)
    blackboard_gray = 255 - blackboard_gray
    #cv2.imshow('', blackboard_gray)
    #cv2.waitKey()
    #print(np.reshape(img, (28, 28)), file)
    #cv2.imshow('', blackboard_gray)
    #cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(blackboard_gray, kernel, iterations=1)
    dilate = crop(dilate)
    new_img = np.zeros((255, 255, 1), np.uint8)


    predicted = quickdraw.classes[quickdraw.keras_predict(quickdraw.model, dilate)[1]]
    if predicted == imgc:
        tp += 1
    else:
        fp += 1
    print(tp, "/", fp, "/", allp, "Predicted:", predicted, "True:", imgc)
    if fp + tp > 10:
        break

print("Accuracy:", tp / (tp + fp) * 100, "%")