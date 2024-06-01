# keras requires  "pip install protobuf==3.20.*" source:
# https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
from abc import ABC

import os
import tensorflow as tf
from matplotlib import pyplot as plt

from keras.models import Model, load_model  # base model class
from keras.layers import Input, Dense, GlobalMaxPooling2D
from keras.applications import VGG16  # VGG16 acts here as base network

from load_aug_img_and_labels_to_tf_dataset import create_final_datasets

# EVIL HACK to remove error #15 when finishing training
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

# CANT RUN THIS ON GPU SO IT DOESN'T MATTER
# limit gpu memory growth
# gpus = tf.config.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# finding face - binary classification, coordinates - regression problem
# download VGG16 (Base architecture) + 1 layer for classification and 1 for regression

# Functional API allows for 2 different loss functions + combining them
# 1 classification model, 1 regression model (localization loss)
# because keras takes in 1 input, 1 output and 1 loss function

# creating an instance of VGG
vgg = VGG16(include_top=False)  # drops everything after last max pooling layer


# to see the layers of VGG16 use:
# print(vgg.summary())
# or
# tracker_model = build_model()
# print(tracker_model.summary())

# Output shape
# (x, -> number of samples
# y, -> width
# z, -> height
# 3) -> number of channels


# build instance of network
def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)  # creating the layer + what to pass to it

    # two output/prediction heads

    # CLASSIFICATION MODEL
    f1 = GlobalMaxPooling2D()(vgg)  # condensing information from vgg layer and returning max values
    class1 = Dense(2048, activation='relu')(f1)  # fully connected layer
    class2 = Dense(1, activation='sigmoid')(class1)  # Dense layer with 1 output
    # sigmoid - value 0-1 - matches 0/1 in classes
    # maps to 0/1 in classification

    # REGRESSION MODEL (BOUNDING BOX MODEL)
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    # maps 4 coordinates of bounding box

    tracker_model = Model(inputs=input_layer, outputs=[class2, regress2])  # Combining both using Model API
    return tracker_model


train, test, val = create_final_datasets()
# print(train.as_numpy_iterator().next()[1])

# Testing out Neural Network
tracker_model = build_model()  # -> creating an instance of build_model()

# see the model
# print(tracker_model.summary())

X, y = train.as_numpy_iterator().next()
# X - images
# y - labels

# print(X.shape)
# returns (8, 120, 120, 3)

classes, coords = tracker_model.predict(X)

# LOSSES AND OPTIMIZERS

# Define optimizer and linear regression

batches_per_epoch = len(train)  # length of joined/final train dataset
lr_decay = (1. / 0.75 - 1) / batches_per_epoch

# https://keras.io/api/optimizers/adam/
# https://faroit.com/keras-docs/0.2.0/optimizers/
# https://arxiv.org/pdf/1412.6980v8.pdf

# adding .legacy. and reinstalling keras seemed to work FUCKING FINALLY
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)  # Adam optimizer


# lr_decay -> how much the learning rate is going to drop each epoch (one entire passing of training data through the
# algorithm)

# loss function - method of evaluating how well the algorith model the data
# loss - penality for bad prediction between prediction and actual

# https://stats.stackexchange.com/questions/319243/object-detection-loss-function-yolo
def localization_loss(y_true, yhat):
    # distance between actual and predicted coordinate (yhat == ŷ)
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    # sum of elements across dimensions of tensor
    # reduce sum into a single outcome

    return delta_coord + delta_size


classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss


# test loss metrics

# print(localization_loss(y[1], coords))
# print(classiffication_loss(y[0], classes))

# add .numpy() for pure value

# TRAINING NEURAL NETWORK

# create custom model class
class TrackerModel(Model):

    # method to pass initial parameters
    def __init__(self, tracker_model, **kwargs):  # ** -> allows for variadic number of elements passed to a function
        # (variable type -> key:value)
        super().__init__(**kwargs)
        self.model = tracker_model

    def compile(self, opt, classloss, localizationloss, **kwargs):  # possible noqa
        super().compile(**kwargs)  # compile() -> computes the Python code from a source object and returns it.
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    # takes in one batch of data and trains on it
    # function for actual training the model
    def train_step(self, batch, **kwargs):
        X, y = batch  # -> unpack batch of data into X and y values
        # X -> preprocessed images

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)  # make a prediction with model (takes in tracker_model)

            batch_classloss = self.closs(y[0], classes)  # passing true and predicted classification values
            # y[0] - true
            # classses - predicted
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)  # cast so loss f works porperly

            total_loss = batch_localizationloss + 0.5 * batch_classloss  # may need tweaking

            # calculating gradient with respect to loss function
            grad = tape.gradient(total_loss, self.model.trainable_variables)

        # gradient descent
        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        # dict
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    # triggered when passing validation dataset
    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


model = TrackerModel(tracker_model)  # Subclass and set up neural network
model.compile(opt, classloss, regressloss)

# train
logdir = 'logs'

# THIS WORKAROUND DOES NOT WORK:
# https://github.com/tensorflow/tensorflow/issues/9512#issuecomment-309143100
# delete log folder each run before creating it and writing a log file to it because tensorflow does not cooperate
# if os.path.exists(logdir):
#     shutil.rmtree(logdir)

# create the folder again for tensorflow to write
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)  # tensorboard for visualizing metrics
hist = model.fit(train, epochs=2, validation_data=val, callbacks=[tensorboard_callback])  # INCREASE TO 10 LATER
# you can do train.take(123) to take in smaller batch

# LIMIT NUMBER OF CORES USED IN PYCHARM TO 7 TO PREVENT CPU OVERLOAD Task manager -> processess -> details
# -> set affinity
# disable a few cores (in my case 1 to bump load down to around 90 - 95%)

# tensorflow does not register GPU and after 4 days I cannot force it to use GPU
# compatibility problem between libraries files, impossible to resolve because of looping missing files and
# tensorflow 2.10 incompatibility with python 3.11

# plot performance
print(hist.history)  # -> prints numerical value for each run in a list

# CRASHED HERE

fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()
# val regressloss should not be jumping

# save the model
tracker_model.save('tracker_model1.h5')
tracker_model = load_model('tracker_model1.h5')

