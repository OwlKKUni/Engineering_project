import os
import cv2
import tensorflow as tf  # build data pipeline
import albumentations as alb
import numpy as np
import json


# DON'T KNOW IF THAT IS NEEDED HERE BUT IT STAYS
# limit GPU memory to tensorflow because the beast is always hungry
def limit_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# BUILD AND RUN AUGMENTATION PIPELINE - for all the images
# Loading up images and labels to augment images - run 1 time
def augment_all_pictures():
    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                             alb.HorizontalFlip(p=0.5),
                             alb.RandomBrightnessContrast(p=0.2),
                             alb.RandomGamma(p=0.2),
                             alb.RGBShift(p=0.2),
                             alb.VerticalFlip(p=0.5)],
                            bbox_params=alb.BboxParams(format='albumentations',
                                                       label_fields=['class_labels']))

    for partition in ['train', 'test', 'validate']:
        for image in os.listdir(os.path.join('../data', partition, 'images')):
            img = cv2.imread(os.path.join('../data', partition, 'images', image))

            coords = [0, 0, 0.00001, 0.00001]  # -> if annotation for image doesn't exist create default annotation
            label_path = os.path.join('../data', partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [640, 480, 640, 480]))

                try:
                    for x in range(60):  # 60 augmented images for each base image
                        augmented = augmentor(image=img, bboxes=[coords],
                                              class_labels=['face'])  # augmentation pipeline
                        cv2.imwrite(os.path.join('../aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                                    augmented['image'])

                        annotation = {}
                        annotation['image'] = image

                        if os.path.exists(label_path):
                            if len(augmented['bboxes']) == 0:
                                annotation['bbox'] = [0, 0, 0, 0]
                                annotation['class'] = 0

                            else:
                                annotation['bbox'] = augmented['bboxes'][0]
                                annotation['class'] = 1

                        else:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0

                        with open(os.path.join('../aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'),
                                  'w') as f:
                            json.dump(annotation, f)  # -> convert python object to JS object


                except Exception as e:
                    print('eeeeeeeeeeeee')


augment_all_pictures()
