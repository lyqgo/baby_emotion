# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:16:30 2019

@author: Administrator
"""


#coding:utf-8

import os, cv2
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use cpu

import numpy as np
import tensorflow as tf
#from PIL import Image
import matplotlib.pyplot as plt
import glob

import model

 

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu 0

 

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

def init_tf(logs_train_dir = './model_use/model.ckpt-10000'):
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

 

def evaluate_image(img_dir):
    # read image

    #im = cv2.imread(img_dir)
    im=img_dir
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (IMG_W, IMG_W))
    image_array = np.array(im)

    prediction = sess.run(pred, feed_dict={x: image_array})
    max_index = np.argmax(prediction)

    pred_label = label_dict_res[str(max_index)]

    print("%s, predict: %s(index:%d), prob: %f" %('.jpg', pred_label, max_index, prediction[0][max_index]))
    #return pred_label, prediction[0][max_index]


if __name__ == '__main__':
    tf.reset_default_graph()

    init_tf()
    #data_path = "/media/DATA2/sku_val"
    '''
    data_path="./datasets/five_val"
    label = os.listdir(data_path)
    for l in label:
        if os.path.isfile(os.path.join(data_path, l)):
            continue
        for img in glob.glob(os.path.join(data_path, l, "*.jpg")):
            evaluate_image(img_dir=img)
    '''
    evaluate_image(img_dir='./datasets/angry.jpg')
    sess.close()
