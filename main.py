import camera
import facialrecognitionmodel
import cv2
import glob

VIDEO_NAME = "test_video"
IMAGE_DIRECTORY = "test_images"
MODEL = "tflite_model.tflite"
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def main():
    video = camera.record_video(VIDEO_NAME)
    camera.extract_frames(video, IMAGE_DIRECTORY)
    model = facialrecognitionmodel.FacialRecognitionModel(MODEL)

    images = glob.glob(IMAGE_DIRECTORY + "/*.jpg") 
    num_success = 0
    num_failed = 0
    for image in images:
        face = facialrecognitionmodel.get_face(image)
        if face is None:
            continue
        recognized = model.recognize(face)
        if recognized:
            num_success += 1
        else:
            num_failed += 1
    percent_success = num_success / (num_success + num_failed)
    if percent_success >= 0.50:
        print("Face recognized")
    else:
        print("Face not recognized")


if __name__ == "__main__":
    main()
