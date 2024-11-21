from PIL import Image
import glob

images = glob.glob('/home/britneyabner/Dropbox/School/Fall2024/ComputerEngineeringLab/FaceRecognition/britney_face/britney/*.jpg')

data_directory = '/home/britneyabner/Dropbox//School/Fall2024/ComputerEngineeringLab/FaceRecognition/formatted_face'

for i, image in enumerate(images):
    img = Image.open(str(image))
    img = img.resize((224, 224))
    file_name = data_directory + '/image_' + str(i) + '.jpg'
    print(file_name)
    img.save(file_name)
