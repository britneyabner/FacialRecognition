import tensorflow as tf
import numpy as np
import glob
import PIL


data_directory = '/home/britneyabner/Dropbox//School/Fall2024/ComputerEngineeringLab/FaceRecognition/britney_face'

VALIDATION_SPLIT = 0.2
IMAGE_HEIGHT = 2944
IMAGE_WIDTH = 2208
BATCH_SIZE = 3

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


def train_convolution():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(
            1./255, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        metrics=['accuracy']
    )

    model.fit(train_ds, val_ds, epochs=5)

    return model


def train_dense():
    model = tf.kera.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ds, val_ds, epochs=5)

    return model



def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    return tflite_model

def write_model(model, name):
    with open(name, 'wb') as f:
        f.write(model)
