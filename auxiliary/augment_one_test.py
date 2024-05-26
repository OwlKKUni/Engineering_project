import os
import cv2
import albumentations as alb
from matplotlib import pyplot as plt
import numpy as np
import json


# APPLY IMAGE AUGMENTATION              - https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

# define augmentatation pipeline
# p - probability of certain augmentation

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='albumentations',  # -> this specific format normalizes values
                                                   label_fields=['class_labels']))

img = cv2.imread(os.path.join('../data', 'train', 'images', '2a52abe3-fd5f-11ed-806b-b42e991176dc.jpg'))

with open(os.path.join('../data', 'train', 'labels', '2a52abe3-fd5f-11ed-806b-b42e991176dc.json'), 'r') as f:
    label = json.load(f)

# extract coordinates and rescale to match image resolution
coords = [0, 0, 0, 0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

# converting pascal coordinates to albumentations coordinates
coords = list(np.divide(coords, [640, 480, 640, 480]))

# apply augmentations
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])

cv2.rectangle(augmented['image'],
              tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)),
              (255, 0, 0), 2)  # -> color and thickness of a rectangle

plt.imshow(augmented['image'])
plt.show()
