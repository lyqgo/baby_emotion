import os, cv2
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu
from rgb import get_face_red
import numpy as np
import tensorflow as tf
#from PIL import Image
import matplotlib.pyplot as plt
import glob
from cv2 import dnn

import model
from statistics import mode
import math
from sklearn.externals import joblib
 
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0

WIDTH = 300
HEIGHT = 300
root = './model_use'
#PROTOTXT = r'F:\Spyder\mobilenetv2_2\model_use\face_detector\deploy.prototxt'
PROTOTXT = './model_use/face_detector/deploy.prototxt'
MODEL = './model_use/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
NET = dnn.readNetFromCaffe(PROTOTXT, MODEL)

def get_facebox(image=None, threshold=0.5):
    """
    Get the bounding box of faces in image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    confidences = []
    faceboxes = []

    NET.setInput(dnn.blobFromImage(
        image, 1.0, (WIDTH, HEIGHT), (104.0, 177.0, 123.0), False, False))
    detections = NET.forward()

    for result in detections[0, 0, :, :]:
        confidence = result[2]
        if confidence > threshold:
            x_left_bottom = int(result[3] * cols)
            y_left_bottom = int(result[4] * rows)
            x_right_top = int(result[5] * cols)
            y_right_top = int(result[6] * rows)
            confidences.append(confidence)
            faceboxes.append(
                [x_left_bottom, y_left_bottom, x_right_top, y_right_top])
    return confidences, faceboxes


label_dict, label_dict_res = {}, {}

with open("./src/label/fer_label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)


N_CLASSES = len(label_dict)
IMG_W = 224
IMG_H = IMG_W 

def init_tf(logs_train_dir = './model_use/model.ckpt-15000'):
    global sess, pred, x
    # process image
    x = tf.placeholder(tf.float32, shape=[IMG_W, IMG_W, 3])
    x_norm = tf.image.per_image_standardization(x)
    x_4d = tf.reshape(x_norm, [-1, IMG_W, IMG_W, 3])
    # predict

    logit = model.MobileNetV2(x_4d, num_classes=N_CLASSES, is_training=False).output
    print("logit", np.shape(logit))

    #logit = model.model4(x_4d, N_CLASSES, is_trian=False)
    #logit = model.model2(x_4d, batch_size=1, n_classes=N_CLASSES)

    pred = tf.nn.softmax(logit)

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    saver.restore(sess, logs_train_dir)
    print('load model done...')

def eva_model(image):
    hog = cv2.HOGDescriptor()
    features = []
    fead = hog.compute(image)
    features.append(fead[:,0].tolist())
    clf = joblib.load(r'F:\Spyder\mobilenetv2_2\model_use\SVC.model')
    return clf.predict(features[0:1])[0]

def evaluate_image(img_dir):
    # read image

    im = cv2.imread(img_dir)   
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (IMG_W, IMG_W))
    image_array = np.array(im)

    flag = eva_model(image_array)
    if flag == 1:
        pred_label = 'finger'
        prob = 0.7
    else:
        prediction = sess.run(pred, feed_dict={x: image_array})
        max_index = np.argmax(prediction)
    
        pred_label = label_dict_res[str(max_index)]
        prob = prediction[0][max_index]
        print("%s, predict: %s(index:%d), prob: %f" %('.jpg', pred_label, max_index, prediction[0][max_index]))#
    return pred_label, prob


if __name__ == '__main__':
    emotion_text0=''
    detection_model_path = './model_use/haarcascade_frontalface_default.xml'
    emotion_model_path = './model_use/model.ckpt-10000'
    frame_window = 10
    emotion_offsets = (0, 0)
    face_cascade = cv2.CascadeClassifier(detection_model_path)
    emotion_window = []
    cv2.namedWindow('window_frame')
    file=['./datasets/00.JPG','./datasets/0.JPG','./datasets/1.JPG','./datasets/2.JPG','./datasets/3.JPG','./datasets/4.JPG','./datasets/5.JPG','./datasets/6.JPG','./datasets/7.JPG','./datasets/8.JPG','./datasets/9.JPG']
    #file=['./datasets/face00.JPG','./datasets/face0.JPG','./datasets/face1.JPG','./datasets/face2.JPG','./datasets/face3.JPG']
    tf.reset_default_graph()
    init_tf()
    while file:
#bgr_image = video_capture.read()[1]
        dir_photo=file.pop()
        bgr_image = cv2.imread(dir_photo)    
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        confidences, faces = get_facebox(rgb_image, threshold=0.5)
        
        
        for (x1,y1,x2,y2) in faces:
            face_coordinates = (x1,y1,x2,y2)
            
            im = cv2.imread(dir_photo)   
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            faces_colour = bgr_image[y1:y2, x1:x2]   #############人脸彩色
            cv2.imwrite('./datasets/temp/1.jpg' ,faces_colour)
            
            cv2.rectangle(rgb_image,(x1,y1),(x2,y2),(0,0,225),1)
            
            red_index=get_face_red(dir_photo,x1,x2,y1,y2)
            if red_index > 120:
                emotion_text0='&warning!'
            else:
                emotion_text0=''
            
            emotion_text,emotion_probability=evaluate_image(img_dir='./datasets/temp/1.jpg')
            label = "%s: %.4f" % (emotion_text+emotion_text0, emotion_probability)
            
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(rgb_image, (x1, y1 - label_size[1]),
                     (x1 + label_size[0],
                      y1 + base_line),
                     (0, 255, 0), cv2.FILLED)
            cv2.putText(rgb_image, label, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            
#            if emotion_text == 'angry':
#                color = emotion_probability * np.asarray((255, 200, 200))
#            elif emotion_text == 'sad':
#                color = emotion_probability * np.asarray((0, 0, 255))
#            elif emotion_text == 'happy':
#                color = emotion_probability * np.asarray((255, 0, 255))
#            elif emotion_text == 'norm':
#                color = emotion_probability * np.asarray((0, 255, 255))
#                
#            else:
#                color = emotion_probability * np.asarray((0, 255, 0))
#    
#            color = color.astype(int)
#            color = color.tolist()
            #draw_bounding_box(face_coordinates, rgb_image, color)
            #draw_text(face_coordinates, rgb_image, emotion_text+emotion_text0,
                      #color, 0, -45, 1, 1)
            
        
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break
sess.close()
cv2.destroyAllWindows()
    
