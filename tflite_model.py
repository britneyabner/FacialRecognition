import tensorflow as tf
import numpy as np
import glob
import PIL
import sys

data_directory = sys.argv[1]

VALIDATION_SPLIT = 0.2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=123,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE
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
        tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3),
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
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
    
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data=val_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)

    return model

def train_resnet50():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.applications.resnet50.ResNet50(
            weights='imagenet'
        )
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data=val_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)

    return model

def train_vgg16(): 
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data=val_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model


def write_model(model, name):
    with open(name, 'wb') as f:
        f.write(model)


if __name__ == "__main__":
    model = train_dense()
    model = convert_to_tflite(model)
    write_model(model, "tflite_model.tflite")
