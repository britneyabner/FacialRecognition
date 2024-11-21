import tensorflow as tf
import cv2

class FacialRecognitionModel:
    def __init__(self, tflite_model):
        self.interpreter = tf.lite.Interpreter(mode_content=tflite_model)
        self.interpreter.allocate_tensor()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        results = self.interpreter.get_tensor(self.output_details[0]['index'])

        return results

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
    else:
        return None



