# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:30:48 2019

@author: Administrator
"""
import cv2
import matplotlib.pyplot as plt

root = './pain.jpg'
crop_size = (224, 224)
img = cv2.imread(root)
plt.imshow(img)
img_new = cv2.resize(img, crop_size)
#plt.imshow(img_new)
