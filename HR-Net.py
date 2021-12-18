import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline

#from tqdm import tqdm_notebook, tnrange
from itertools import chain
#from skimage.io import imread, imshow, concatenate_images
#from skimage.transform import resize
#from skimage.morphology import label
#from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_im
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, merge

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Convolution2D
from keras.models import load_model
import os
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
from keras.utils import np_utils
import os.path
import h5py
import keras
import time
import numpy as np
from scipy import misc
import sklearn
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

from keras.models import load_model
import numpy
import sklearn
from keras.layers import Input, Convolution2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

from keras.models import load_model
import numpy
import sklearn

#import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D,BatchNormalization
#import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
import scipy.io as sio
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Set some parameters
im_width = 128
im_height = 128
border = 5
#path_train = r'D:\job\LAL\data\data\Figures_Only_400-20210625T150802Z-001\deep learning\training/'
#path_train = r'D:\job\LAL\data\data\400_figures\deep-learning/'
#path_train= r'D:\job\LAL\data\data\400_figures\deep-learning\deep-learning-training-data1/'
path_train = r'D:\job\LAL\data\Figures_Only-20210704T024044Z-001\deep learning\training/'
# Get and resize train images and masks
def get_data(path, train=True):
    ids = next(os.walk(path + "images"))[2]
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X


X_train, y_train = get_data(path_train, train=True)

# Check if training data looks all right
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(X_train[ix, ..., 0], cmap='seismic', interpolation='bilinear')
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])
ax[0].set_title('Input Images (dummy color)')

ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
ax[1].set_title('Ground-truth');



def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),  kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def HRnet3(data,  n_filters=16, dropout=0.1, batchnorm=True):
    ## high resolution
    c1ht1 = conv2d_block(data, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c12ht12 = conv2d_block(c1ht1, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    #c13ht13 = conv2d_block(c12ht12, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    #c14ht14 = conv2d_block(c13ht13, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    #c15ht15 = conv2d_block(c14ht14, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    c2d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c12ht12)
    ## Mid resolution
    p1 = MaxPooling2D((2, 2))(c12ht12)
    p1 = Dropout(dropout)(p1)
    c22 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c22u = UpSampling2D()(c22)
    c2h = concatenate([c12ht12, c22u])
    c22m = concatenate([c22, c2d])

    c3 = conv2d_block(c2h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c33 = conv2d_block(c22m, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c3d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c3)
    c33u = UpSampling2D()(c33)
    c3h = concatenate([c3, c33u])
    c33m = concatenate([c33, c3d])

    p2 = MaxPooling2D((2, 2))(c33)
    p2 = Dropout(dropout)(p2)
    c444 = conv2d_block(p2, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c3dd = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c3d)
    c444l = concatenate([c444, c3dd])

    c4 = conv2d_block(c3h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c44 = conv2d_block(c33m, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c444 = conv2d_block(c444l, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    c4d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c4)
    c4dd = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c4d)
    c44u = UpSampling2D()(c44)
    c44d = Conv2D(n_filters * 1, kernel_size=1, strides=2)(c44)
    c444u = UpSampling2D()(c444)
    c444uu = UpSampling2D()(c444u)

    c5h = concatenate([c4, c44u, c444uu])
    c55m = concatenate([c44, c4d, c444u])
    c555l = concatenate([c444, c44d, c4dd])

    c5 = conv2d_block(c5h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    c55 = conv2d_block(c55m, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    c555 = conv2d_block(c555l, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    c555u = UpSampling2D()(c555)
    c555uu = UpSampling2D()(c555u)
    c55u = UpSampling2D()(c55)
    c6h = concatenate([c5, c55u, c555uu])

    c6 = conv2d_block(c6h, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)
    model = Model(inputs=[data], outputs=[outputs])
    return model


input_img = Input((im_height, im_width, 1), name='img')
model = HRnet3(input_img, n_filters=16, dropout=0., batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 2018
bs = 32

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)

# Just zip the two generators to get a generator that provides augmented images and masks at the same time
train_generator = zip(image_generator, mask_generator)

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
    ModelCheckpoint('model-tgs-hrnet.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit_generator(train_generator, steps_per_epoch=(len(X_train) // bs), callbacks=callbacks, epochs=100,
                              validation_data=(X_train, y_train))

model.save("model-hrnet-new1.h5")

path_test = r'D:\job\LAL\data\data\400_figures\deep-learning\deep-learning-training-data1\testing-data/'
X_valid, y_valid = get_data(path_test, train=True)

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)

# Predict on train, val and test
#preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Original-Dummy-colors')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Ground-truth Mask')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Mask Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Predicted binary');

def plot_sample1(X, y, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(y[ix].squeeze())
    ax[0].set_title('Ground-truth Mask')
    ax[1].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Predicted binary');

#plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=2)
plot_sample1(X_valid, y_valid, preds_val_t, ix=2)
