import cv2
import glob
import numpy as np

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 900

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')


def get_face(image):
    image_read = cv2.imread(str(image))
    image_read = cv2.resize(image_read, (IMAGE_WIDTH, IMAGE_HEIGHT))
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


def train(image_directory, name):
    name_dict = {name: 0}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = glob.glob(image_directory + "/*.jpg")
    faces = []
    labels = []
    for image in images:
        face = get_face(image)
        if face is not None:
            faces.append(face)
            labels.append(name_dict[name])

    recognizer.train(faces, np.array(labels))

    return recognizer

if __name__ == "__main__":
    model = train("/home/britneyabner/Dropbox/School/Fall2024/ComputerEngineeringLab/FaceRecognition/britney_face/britney", "Britney")
    model.save('opencv_model.xml')
