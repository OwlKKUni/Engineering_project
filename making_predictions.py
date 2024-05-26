# MAKING PREDICTIONS on test set

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model  # base model class

from load_aug_img_and_labels_to_tf_dataset import create_final_datasets

os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'  # EVIL HACK 2 : Electric Boogaloo

train, test, val = create_final_datasets()
facetracker = load_model('facetracker.h5', compile=False)  # Another hack from SO:
# https://stackoverflow.com/questions/53295570/userwarning-no-training-configuration-found-in-save-file-the-model-was-not-c

test_data = test.as_numpy_iterator()  # set up an iterator
test_sample = test_data.next()  # grab next batch (8)
yhat = facetracker.predict(test_sample[0])  # run a prediction

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.5:  # If classification loss is > 0.5
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

    ax[idx].imshow(sample_image)

plt.show()
