import picamera2
import cv2
import os
import pathlib


def record_video(file_name: str) -> pathlib.Path:
    # record a video with Raspberry Pi camera module and write to file
    file_name = file_name + ".mp4"
    camera = picamera2.Picamera2()
    camera_config = camera.create_preview_configuration()
    camera.configure(camera_config)
    camera.start_and_record_video(file_name, duration=5)

    # return the updated file name with the file extension
    return file_name


def extract_frames(video: str, output_directory: str):
    # make output directory if it doesn't exist yet
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # extract the frames from the video and write to output directory
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(output_directory, '%d.jpg') % count, image)
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
