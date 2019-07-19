# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:27:39 2019

@author: 84270
"""

import cv2

photo_dir='./datasets/0.JPG'

def get_face_red(photo_dir,x1,x2,y1,y2):
    x1=max(x1, int(1.1*x1+1))
    x2=min(int(0.9*x2-1), x2)
    y1=max(y1, int(1.1*y1+1))
    y2=min(y2,int(0.9*y2-1))
    im = cv2.imread(photo_dir)   
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    num=0
    for a in range(x1,x2-1):
        for b in range(y1,y2-1):
            num=num+im[a,b][0]*2-im[a,b][1]-im[a,b][2]
    mean=num/((x2-x1)*(y2-y1))
    print('mean',mean)
    return mean

if __name__ == '__main__':
    get_face_red(photo_dir,21,61,81,100)
    
'''
x1,x2,y1,y2=21,61,81,100
im = cv2.imread(photo_dir)   
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
num=0
for a in range(x1,x2):
    for b in range(y1,y2):
        num=num+im[a,b][0]
mean=num/((x2-x1)*(y2-y1))
print('mean',mean)
'''
