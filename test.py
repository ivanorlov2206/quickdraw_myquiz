import os
import quickdraw_api

classes = ['book', 'sun', 'banana', 'apple', 'bowtie', 'ice cream', 'eye', 'square', 'door', 'sword', 'star', 'fish', 'bucket', 'donut', 'mountain']
classes = sorted(classes)
data_folder = '/home/mineorpe/work/restests/data/'
files = os.listdir(data_folder)
fp = 0
tp = 0
allp = len(files)
for file in files:
    imgc = classes[int(file.split("(")[0])]
    predicted = quickdraw_api.keras_predict(quickdraw_api.model, data_folder + file)
    if predicted == imgc:
        tp += 1
    else:
        fp += 1
    print(tp + fp, "/", allp)
print("Accuracy:", tp / allp * 100, "%")