# -*- coding: utf-8 -*-
"""
Created on Sun July 20 22:19:53 2021
©2020. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
@author: Reshad
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

data = r"D:\job\LAL\data\data-and-result\data\Figures_Only-400\Figures_Only/" ### Input Images locations
data_list = os.listdir(data)
result_location = r'D:\job\LAL\data\data-and-result\results\point-shooting-400-with-roi/' ## Output images
count = 0
ROI_number=0
start_time = time.time()
for i in range(len(data_list)):
    print(count)
    os.chdir(data)
    # Load image, grayscale, blur, Otsu's threshold
    img = cv2.imread(data_list[i])  ### reading image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  ### gray conversion
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  ## making 0 or 255 pixel
    ### Black pixel co-ordinate detection
    ii1 = np.nonzero(thresh == 255)
    x = list(ii1[1])
    y = list(ii1[0])
    # Now, loop through coord arrays, and create a circle at each x,y pair
    os.chdir(result_location)
    for xx, yy in zip(x, y):
        cv2.circle(img, (xx, yy), 10, (0, 20, 200), 10)
        # cv2.imwrite('output_38.png',img)
    cv2.imwrite("output_%d.png" % count, img)  ### Saving Masked Image

    os.chdir(data)
    img = cv2.imread(data_list[i])
    os.chdir(result_location)
    #cv2.imwrite("original_%d.png" % count, img)  ## Saving Original Image
    img1 = cv2.imread("output_%d.png" % count)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    canny_get_edge = cv2.Canny(gray, 40, 250)
    ################# Morphology Operation ########################
    # Perform a little bit of morphology:
    # Set kernel (structuring element) size:
    kernelSize = (3, 3)
    # Set operation iterations:
    opIterations = 1
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    # Perform Dilate:
    canny_get_edge = cv2.morphologyEx(canny_get_edge, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                                      cv2.BORDER_REFLECT101)
    ######## Contours Detection ################
    contours, hierarchy = cv2.findContours(canny_get_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    cv2.drawContours(img, contours, -1, (0, 255, 0), 4)
    cv2.imshow('Contours', img)
    #cv2.imwrite("result_%d.png" % count, img)
    count = count + 1
    os.chdir(data)
    im = cv2.imread(data_list[i]) ### reading image
    copy=im.copy()
    os.chdir(result_location)
    ################ Drawing Bounding Box ####################
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50: continue
        cv2.contourArea(c)
        x, y, w, h = rect
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ROI = im[y:y + h, x:x + w]
        cv2.imwrite('ROI_{}_{}.png'.format(i, ROI_number), ROI)
        ROI_number=ROI_number+1
    cv2.imwrite("result_bounding_box_%d.png" % count, copy)
    ROI_number = 0
    #os.remove("output_%d.png")
print('Total Time of Point-shooting')
print("--- %s seconds ---" % (time.time() - start_time))
