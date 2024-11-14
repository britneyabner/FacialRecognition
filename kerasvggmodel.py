import tensorflow as tf
import keras
import keras_vggface

vggface = keras_vggface.VGGFace(model='vgg16')

print(vggface.summary())
