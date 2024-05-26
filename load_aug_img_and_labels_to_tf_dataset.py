# https://www.w3schools.com/python/python_lambda.asp -> lambda expression
# https://www.geeksforgeeks.org/python-map-function/ -> map function
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import cv2
import numpy as np


def load_image(x):  # x - full file path
    byte_img = tf.io.read_file(x)  # byte encoded image from a filepath
    img = tf.io.decode_jpeg(byte_img)  # get the image back
    return img


# gets full filepath name
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']  # -> returns two arrays


def image_labels_operations():
    # this instead of *.jpg works !!!
    # shuffle as false because of the labels
    train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
    train_images = train_images.map(lambda x: x / 255)  # -> gives value between 0 - 1
    # so we can use it for sigmoid activation function

    test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
    test_images = test_images.map(lambda x: x / 255)

    validate_images = tf.data.Dataset.list_files('aug_data\\validate\\images\\*.*.jpg', shuffle=False)
    validate_images = validate_images.map(load_image)
    validate_images = validate_images.map(lambda x: tf.image.resize(x, (120, 120)))
    validate_images = validate_images.map(lambda x: x / 255)

    train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    validate_labels = tf.data.Dataset.list_files('aug_data\\validate\\labels\\*.json', shuffle=False)
    validate_labels = validate_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    # lambda will loop through every filename in a folder
    # integer on 8 bits

    # display loaded stuff
    # print(train_labels.as_numpy_iterator().next())  # -> view content
    # print(validate_images.as_numpy_iterator().next())

    # print(len(train_images), len(train_labels), len(test_images),
    #      len(test_labels), len(validate_images), len(validate_labels))
    # returns 5220 5220 1140 1140 1200 1200

    return train_images, train_labels, test_images, test_labels, validate_images, validate_labels


# COMBINE LABEL AND IMAGE SAMPLES


# Create final datasets
def create_final_datasets():
    train_images, train_labels, test_images, test_labels, validate_images, validate_labels = image_labels_operations()

    train = tf.data.Dataset.zip((train_images, train_labels))  # -> zips (combines) together datasets
    # (creates this type of generator)
    train = train.shuffle(7000)  # -> bigger than the size of dataset
    train = train.batch(8)  # -> each batch - 8 images / 8 labels
    train = train.prefetch(4)  # -> helps eliminating bottlenecks

    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(1700)
    test = test.batch(8)
    test = test.prefetch(4)

    val = tf.data.Dataset.zip((validate_images, validate_labels))
    val = val.shuffle(1500)
    val = val.batch(8)
    val = val.prefetch(4)

    # print(train.as_numpy_iterator().next()[0].shape) # for images
    # prints (8, 120, 120, 3) -> 8 images 120x120 pixels and 3 color channels

    # print(train.as_numpy_iterator().next()[1]) # for labels
    # returns all the classes and bounding boxes coords in an array

    return train, test, val


# view images and annotations
def view_images():
    train, test, val = create_final_datasets()

    data_samples = train.as_numpy_iterator()
    res = data_samples.next()  # -> grabs next batch of images

    # prints images with bounding boxes
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]

        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

        ax[idx].imshow(sample_image)

    plt.show()
