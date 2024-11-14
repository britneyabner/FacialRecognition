import face_recognition
import pathlib
import pickle
import sys


def _encode_faces(name: str, directory: str) -> list:
    for filepath in pathlib.Path(directory).glob("*.jpg"):
        names = []
        encodings = []

        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model="hog")
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            encodings.append(encoding)
            names.append(name)

    name_encodings = {"names": names, "encodings": encodings}

    return name_encodings


def train(name: str, directory: str) -> None:
    encodings = _encode_faces(name, directory)
    with open(f"{name}.pkl", 'wb') as file:
        pickle.dump(encodings, file)


def _recognize_face(unknown_encoding, loaded_encodings) -> bool:
    matches = face_recognition.compare_faces(loaded_encodings["encodings"],
                                             unknown_encoding)

    num_matches = 0
    for match in matches:
        if match:
            num_matches += 1

    match_ratio = num_matches / len(matches)

    if match_ratio >= 0.50:
        return True
    else:
        return False


def recognize(image_path: str, encoding_path: str):
    encoding_path = pathlib.Path(encoding_path)
    with encoding_path.open(mode='rb') as file:
        encodings = pickle.load(file)

    known_encoding = encodings

    image = face_recognition.load_image_file(image_path)

    face_encodings = face_recognition.face_encodings(image)
    
    is_recognized = face_recognition.compare_faces(known_encoding, face_encodings)
    if is_recognized:
        print("Face recognized")
    else:
        print("Face not recognized")


def run_training():
    name = sys.argv[1]
    directory = sys.argv[2]
    train(name, directory)


def run_recognizing():
    image_path = sys.argv[1]
    encoding_path = sys.argv[2]

    recognize(image_path, encoding_path)


if __name__ == "__main__":
    run_recognizing()
