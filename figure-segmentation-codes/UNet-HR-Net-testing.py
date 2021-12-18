# -*- coding: utf-8 -*-
"""
Created on Sun July 20 22:19:53 2021
©2020. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
@author: Reshad
"""
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
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from skimage.io import imread_collection

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from tensorflow import keras
import time
os.chdir(r'D:\job\LAL\data\Figures_Only-20210704T024044Z-001\deep learning')
#model = keras.models.load_model('model-unet-new1-without-tagg-400.h5') ## Unet
model=  keras.models.load_model('model-hrnet-new1.h5') ## Unet

# Set some parameters
im_width = 128
im_height = 128
border = 5


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

#path_test = r'D:\job\LAL\data\Figures_Only-20210704T024044Z-001\deep learning\testing\testing-2/'
path_test = r'D:\job\LAL\data\data-and-result\test-paper\orginal/'
X_valid, y_valid = get_data(path_test, train=True)

preds_val = model.predict(X_valid, verbose=1)

# Threshold predictions
#preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


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

plot_sample1(X_valid, y_valid, preds_val_t, ix=2)

orginal_path = r'D:\job\LAL\data\data-and-result\test-paper\orginal\images/*.jpg'
X_or= imread_collection(orginal_path)

#os.chdir(r'D:\job\LAL\data\data-and-result\test-paper\unet/')
for i in range(len(X_valid)):
    print('Image is going to processed : ', i)
    img = X_valid[i].copy()
    preds=preds_val[i]
    preds=cv2.normalize(src=preds, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blur = cv2.GaussianBlur(preds, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #canny_get_edge= cv2.Canny(thresh,40,250)
    # Perform a little bit of morphology:
    # Set kernel (structuring element) size:
    kernelSize = (3, 3)
    # Set operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    # Perform Dilate:
    morphology = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)
    contours, hierarchy = cv2.findContours(morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 4)
    #plt.imshow(img)
    im=X_or[i].copy()
    orx = im.shape[1]
    ory = im.shape[0]
    scalex = orx / 128
    scaley = ory / 128
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 5 or rect[3] < 5: continue
        cv2.contourArea(c)
        x, y, w, h = rect
        x = int(x * scalex)
        y = int(y * scaley)
        w = int(w * scalex)
        h = int(h * scaley)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
    im = cv2.normalize(src=im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    os.chdir(r'D:\job\LAL\data\data-and-result\test-paper\unet-bb\h-2/')
    im = cv2.normalize(src=im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("result_bounding_box_%d.png" % i, im)

