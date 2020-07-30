import numpy as np
import os
import pickle
import sys

# Put everything to data directory
files = os.listdir("data/")

x_load = []
y_load = []

def load_data():
    count = 0
    bts = 0
    for file in files:
        file = "data/" + file
        x = np.load(file)
        x = x[0:10000, :]
        print(x[0])
        x = x.astype('float32') / 255.
        bts += sys.getsizeof(x)
        x_load.append(x)
        y = [count for _ in range(10000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)
        print(file, bts)

    return x_load, y_load

def load_ex():
    count = 0
    bts = 0
    for file in files:
        file = "data/" + file
        x = np.load(file)
        x = x[10000:20000, :]
        x = x.astype('float32') / 255.
        bts += sys.getsizeof(x)
        x_load.append(x)
        y = [count for _ in range(10000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)
        print(file, bts)

    return x_load, y_load

print(files)
features, labels = load_data()
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')
features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])


with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)

features2, labels2 = load_ex()
features2 = np.array(features2).astype('float32')
labels2 = np.array(labels2).astype('float32')
features2 = features2.reshape(features2.shape[0] * features2.shape[1], features2.shape[2])
labels2 = labels2.reshape(labels2.shape[0] * labels2.shape[1], labels2.shape[2])

with open("features_ex", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels_ex", "wb") as f:
    pickle.dump(labels, f, protocol=4)

print(1)
