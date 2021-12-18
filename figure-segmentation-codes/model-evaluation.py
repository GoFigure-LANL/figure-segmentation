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

#data = r"D:\job\LAL\data\data-and-result\data\400_figures\400_figures\results\Figure_only_png-20211113T023414Z-001\Figure_only_png"  ### Input Images locations
data= r'D:\job\LAL\data\data-and-result\data\400_figures\400_figures\results\testing-images-100-from-400/'
data_list = os.listdir(data)
#result_location = r'D:\job\LAL\data\data-and-result\data\400_figures\400_figures\results\result\point-shooting-results\100/'
result_location = r'D:\job\LAL\data\data-and-result\data\400_figures\400_figures\results\result\point-shooting-results\100-400/'  ## Output images
count = 0
ROI_number = 0
start_time = time.time()
all_pred_coordinates={}
predicted_coordinates=[]
names=[]
for i in range(len(data_list)):
    name= data_list[i]
    names.append(name)
    print(count)
    print(name)
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
    # cv2.imwrite("original_%d.png" % count, img)  ## Saving Original Image
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
    #cv2.imshow('Contours', img)
    # cv2.imwrite("result_%d.png" % count, img)
    count = count + 1
    os.chdir(data)
    im = cv2.imread(data_list[i])  ### reading image
    copy = im.copy()
    os.chdir(result_location)
    ################ Drawing Bounding Box ####################
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50: continue
        cv2.contourArea(c)
        x, y, w, h = rect
        tuple2 = (x, y, w, h)
        predicted_coordinates.append(tuple2)
        cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ROI = im[y:y + h, x:x + w]
        #cv2.imwrite('ROI_{}_{}.png'.format(i, ROI_number), ROI)
        ROI_number = ROI_number + 1
    cv2.imwrite("result_bounding_box_%d.png" % count, copy)
    ROI_number = 0
    # os.remove("output_%d.png")
    all_pred_coordinates[name]=predicted_coordinates
    predicted_coordinates=[]

import pandas as pd
os.chdir(r'D:\job\LAL\data\data-and-result\data\400_figures\400_figures\annotate_bbox_image\annotate_bbox_image')
cat_df = pd.read_json('annotate_patentimage_json.json')
#os.chdir(r'D:\job\LAL\data\data-and-result\data\400_figures\400_figures\results\testing_Images_100')
#cat_df = pd.read_json('via_project_11Nov2021_17h15m_json.json')
cat_df_modify = cat_df.transpose().reset_index()[['filename', 'regions']]
coordinates={}
coordinate=[]
for i in range(len(names)):
    element = cat_df_modify[cat_df_modify['filename'] == names[i]]
    a = element.index
    a = a[0]
    size = len(element['regions'][a])
    for j in range(size):
        current = element['regions'][a][j]['shape_attributes']
        tupple1 = (current['x'], current['y'], current['width'], current['height'])
        coordinate.append(tupple1)
    coordinates[names[i]]=coordinate
    coordinate=[]


def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
    return I / U


def iou_pred(y_test, y_pred):
    z = len(y_test)
    z1 = len(y_pred)
    result = []
    for i in range(len(y_test)):
        for j in range(len(y_pred)):
            result.append(IOU(y_test[i], y_pred[j]))
    results = sorted(result, reverse=True)
    if len(y_test) == len(y_pred):
        fresult = results[:z]
    else:
        fresult = results[:z1]
    p1 = [1 if i > .7 else 0 for i in fresult]
    if z!=z1:
        return 0
    elif sum(p1) / len(p1) == 1:
        return 1
    else:
        return 0

total_result=0
for i in range(len(names)):
    total_result=total_result + iou_pred(coordinates[names[i]], all_pred_coordinates[names[i]])

print('Total correct number', total_result)
