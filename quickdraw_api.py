import threading
import time

import cv2
from keras.applications import mobilenet, imagenet_utils
from keras.models import load_model
import numpy as np
from keras.applications.mobilenet import preprocess_input
import keras.preprocessing.image

model = load_model('model.h5')

classes = ['book', 'sun', 'banana', 'apple', 'bowtie', 'ice cream', 'eye', 'square', 'door', 'sword', 'star', 'fish', 'bucket', 'donut', 'mountain']
classes = sorted(classes)

def classif(fname):
    # ------------ image preprocessing ---------------------
    digit2 = cv2.imread(fname)
    blackboard_gray = cv2.cvtColor(digit2, cv2.COLOR_BGR2GRAY)
    blackboard_gray2 = 255 - blackboard_gray
    blur1 = cv2.medianBlur(blackboard_gray2, 15)
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
            cv2.imwrite(fname, digit)
            pred_class = keras_predict(model, fname)
    return pred_class



def keras_predict(model, fname):
    img = keras.preprocessing.image.load_img(fname)
    img = img.resize((224, 224))
    doc = keras.preprocessing.image.img_to_array(img)
    print(doc.shape)
    doc = np.expand_dims(doc, axis=0)
    doc = preprocess_input(doc)
    predictions = model.predict(doc)
    print(predictions)
    results = np.argmax(predictions)
    return classes[results]



keras_predict(model, 'test.png')


if __name__ == '__main__':
    print(classif("118858458432493.png"))
    # 1 thread 18.429327726364136 seconds
    # 2 threads 11.09047245979309 seconds
    # 3 threads 9.253944873809814 seconds
    # 4 threads 7.361640453338623