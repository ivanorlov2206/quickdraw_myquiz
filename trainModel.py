import numpy as np
from keras import Input, Model
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D
from keras.callbacks import TensorBoard

BATCH_SIZE = 15
classes = ['book', 'sun', 'banana', 'apple', 'bowtie', 'ice cream', 'eye', 'square', 'door', 'sword', 'star', 'fish', 'bucket', 'donut', 'mountain']



def main():
    base_model = MobileNet(weights="imagenet", include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    preds = Dense(15, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=preds)
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory("./data/", target_size=(224, 224), color_mode='rgb', batch_size=8, class_mode='categorical', shuffle=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    step_size_train = train_generator.n // train_generator.batch_size
    model.fit_generator(train_generator, steps_per_epoch=step_size_train, epochs=5)

    model.save("model.h5")

main()
