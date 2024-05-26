# Get number of images in a folder
import os
import time
import uuid  # uniform unique identifier
import cv2  # Open CV - for camera FORCE REINSTALLED TO 4.5.5.62


def get_number_of_images():
    folder_names = ['data/images', 'data/test/images', 'data/train/images', 'data/validate/images']
    totalFiles = 0
    for name in folder_names:
        APP_FOLDER = name
        for base, dirs, files in os.walk(APP_FOLDER):
            for Files in files:
                totalFiles += 1
    return totalFiles


# SET-UP AND GETTING DATA - TAKING PICTURES

# Pictures are in 640 x 480 resolution
def take_pictures():
    total_images = get_number_of_images()
    if total_images < 120:
        IMAGES_PATH = os.path.join('data', 'images')
        number_images = 30  # Glasses, no glasses etc / get some pictures out of frame for negative samples

        # collect images
        capture = cv2.VideoCapture(0)  # this has to be 0 instead of 1 (because I have only one webcam)
        for imgnum in range(number_images):
            print('Collecting image {}'.format(imgnum))  # places imgnum inside curly brackets

            ret, frame = capture.read()  # boolean value 1 if frame read correctly
            # store the array values of respective pixels of the video

            imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgname, frame)
            cv2.imshow('frame', frame)
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


take_pictures()
