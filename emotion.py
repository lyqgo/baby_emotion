# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:47:32 2018

@author: 84270
"""

from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import math

#from utils.datasets import get_labels
#from utils.inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets
#from utils.inference import load_detection_model
#from utils.preprocessor import preprocess_input
#from utils.eye_rotation import eyes_1
#from utils.eye_rotation import rotation

# parameters for loading data and images
detection_model_path = './model_use/haarcascade_frontalface_default.xml'
#eye_model_path = '../models/haarcascade_eye.xml'
emotion_model_path = './model_use/model.ckpt-10000'
#emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
#face_detection = load_detection_model(detection_model_path)
face_cascade = cv2.CascadeClassifier(detection_model_path)
#eye_cascade = cv2.CascadeClassifier(eye_model_path)    ##############
#emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
#emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
#video_capture = cv2.VideoCapture('../1.mp4')
while True:
    #bgr_image = video_capture.read()[1]
    bgr_image = cv2.imread('./datasets/laugh.JPG')    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_image)   #####获得人脸
    #print(faces)
    
    for (x,y,w,h) in faces:
        face_coordinates = (x,y,w,h)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        faces_colour = bgr_image[y1:y2, x1:x2]   #############人脸彩色
        cv2.rectangle(bgr_image,(x,y),(x+w,y+h),(0,0,255),2)
        
        emotion_text,emotion_probability=evaluate_image(faces_colour)
        #emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
    