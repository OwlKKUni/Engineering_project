# NOT USED ANYWHERE, MORE LIKE EXERCISE

# load image into TF Data Pipeline
import tensorflow as tf
from matplotlib import pyplot as plt


def load_image(x):  # x - full file path
    byte_img = tf.io.read_file(x)  # byte encoded image from a filepath
    img = tf.io.decode_jpeg(byte_img)  # get the image back
    return img

# commented because bugs load_..._to_dataset.py for some reason...

# images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False)  # anything in jpg format
# images = images.map(load_image)
#
# # VISUALISE IMAGES IN MATPLOTLIB             # Returns an iterator which converts all elements of the dataset to numpy
#
# image_generator = images.batch(4).as_numpy_iterator()  # batch so that it returns 4 instead of 1 value
# plot_images = image_generator.next()
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image)
# plt.show()
