import os

import cv2
import numpy as np

import quickdraw

classes = ['book', 'sun', 'banana', 'apple', 'bowtie', 'ice cream', 'eye', 'square', 'door', 'sword', 'star', 'fish', 'bucket', 'donut', 'mountain']
classes = sorted(classes)
data_folder = '/home/mineorpe/work/restests/data/'
files = os.listdir(data_folder)
fp = 0
tp = 0
allp = len(files)
for file in files:
    imgc = classes[int(file.split("(")[0])]
    fname = data_folder + file
    digit2 = cv2.imread(fname)
    blackboard_gray = cv2.cvtColor(digit2, cv2.COLOR_BGR2GRAY)
    blackboard_gray = 255 - blackboard_gray
    image_x = 28
    image_y = 28
    img = cv2.resize(blackboard_gray, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = img / 255.
    img = np.where(img, 1, 0)
    print(np.reshape(img, (28, 28)), file)
    #cv2.imshow('', blackboard_gray)
    #cv2.waitKey()
    predicted = quickdraw.classes[quickdraw.keras_predict(quickdraw.model, blackboard_gray)[1]]
    print(predicted, imgc)
    if predicted == imgc:
        tp += 1
    else:
        fp += 1
    print(tp + fp, "/", allp)

print("Accuracy:", tp / allp * 100, "%")