import tensorflow as tf
import pathlib
import cv2
import numpy as np

class FacialRecognitionModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite
        self.interpreter = tf.lite.interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def recognize(self, image) -> bool:
        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self.output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])
        probabilities = tf.nn.softmax(probabilities)
        largest_prob = tf.math.argmax(probabilities)
        
        if probabilities[largest_prob] >= 0.75:
            return True
        else:
            return False


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')
def get_face(image, width, height):
    image_read = cv2.imread(str(image))
    image_read = cv2.resize(image_read, (width, height))
    gray = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=5,
                                                   minSize=(300, 300))
    if len(detected_faces) == 1:
        face_crop = gray[detected_faces[0][1]:detected_faces[0][1] +
                         detected_faces[0][3],
                         detected_faces[0][0]:detected_faces[0][0] +
                         detected_faces[0][2]]

        return face_crop
    return None



