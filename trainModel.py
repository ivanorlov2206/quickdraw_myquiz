import numpy as np
from keras import Input, Model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, add
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import TensorBoard

BATCH_SIZE = 15


stride = 1
CHANNEL_AXIS = 3

def res_layer(x, filters, pooling=False, dropout=0.0):
    temp = x
    temp = Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)
    temp = BatchNormalization(axis=CHANNEL_AXIS)(temp)
    temp = Activation("relu")(temp)
    temp = Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)

    x = add([temp, Conv2D(filters, (3, 3), strides=stride, padding="same")(x)])
    if pooling:
        x = MaxPooling2D((2, 2))(x)
    if dropout != 0.0:
        x = Dropout(dropout)(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    return x



def keras_model(image_x, image_y):
    num_of_classes = BATCH_SIZE
    inp = Input(shape=(28, 28, 1))
    x = inp
    x = Conv2D(16, (3, 3), strides=stride, padding="same")(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    x = res_layer(x, 32, dropout=0.2)
    x = res_layer(x, 32, dropout=0.3)
    x = res_layer(x, 32, dropout=0.4, pooling=True)
    x = res_layer(x, 32, dropout=0.2)
    x = res_layer(x, 32, dropout=0.2, pooling=True)
    x = res_layer(x, 256, dropout=0.4)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.23)(x)
    x = Dense(num_of_classes, activation="softmax")(x)
    model = Model(inp, x, name="Resnet")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "QuickDraw.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def prepress_labels(labels):
    labels = np_utils.to_categorical(labels)
    return labels


def get_eval(y_true, y_prob):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    output['loss'] = metrics.log_loss(y_true, y_prob)
    output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output

def main():
    model, callbacks_list = keras_model(28, 28)
    print_summary(model)
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    labels=prepress_labels(labels)
    print(labels.shape)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=1,
                                                        test_size=0.2)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.25, random_state=1)
    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
    valid_x = valid_x.reshape(test_x.shape[0], 28, 28, 1)
    model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=25, batch_size=64,
              callbacks=[TensorBoard(log_dir="QuickDraw")])
    res = model.evaluate(test_x, test_y, batch_size=64)

    print("Loss, acc:", res)
    model.save('QuickDraw.h5')


main()
