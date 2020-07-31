import numpy as np
import os
import pickle

from PIL import Image
from quickdraw import QuickDrawDataGroup

import sys

# Put everything to data directory
classes = ['book', 'sun', 'banana', 'apple', 'bowtie', 'ice cream', 'eye', 'square', 'door', 'sword', 'star', 'fish', 'bucket', 'donut', 'mountain']


def load_data():
    for cl in classes:
        #os.mkdir("data/" + cl)
        drawings = QuickDrawDataGroup(cl).drawings
        count = 0
        for drawing in drawings:
            imgn = drawing.image.resize((224, 224), Image.ANTIALIAS)
            imgn.save("data/" + cl + "/" + cl + "({}).png".format(str(count)))
            count += 1
            if count >= 200:
                break


load_data()
print(1)
