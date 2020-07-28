import threading
import time

import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os
from PIL import Image

model = load_model('QuickDraw.h5')

classes = ['book.npy', 'sun.npy', 'banana.npy', 'apple.npy', 'bowtie.npy', 'ice cream.npy', 'eye.npy', 'square.npy', 'door.npy', 'sword.npy', 'star.npy', 'fish.npy', 'bucket.npy', 'donut.npy', 'mountain.npy']


def classif(fname):
    # ------------ image preprocessing ---------------------
    digit2 = cv2.imread(fname)
    blackboard_gray = cv2.cvtColor(digit2, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.medianBlur(blackboard_gray, 15)
    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
    thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # -------------- image segmentation----------------------
    blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    if len(blackboard_cnts) >= 1:
        cnt = max(blackboard_cnts, key=cv2.contourArea)
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            digit = blackboard_gray[y:y + h, x:x + w]
            pred_probab, pred_class = keras_predict(model, digit)
    return classes[pred_class]



def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img



keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))


if __name__ == '__main__':
    print(classif("test.png"))
    # 1 thread 18.429327726364136 seconds
    # 2 threads 11.09047245979309 seconds
    # 3 threads 9.253944873809814 seconds
    # 4 threads 7.361640453338623