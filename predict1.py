# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:23:36 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np

sess = tf.Session()
X = None # input
yhat = None # output

def load_model():
    """
        Loading the pre-trained model and parameters.
    """
    global X, yhat
    #modelpath = r'/home/senius/python/c_python/test/'
    saver = tf.train.import_meta_graph('./model_save/model.ckpt-5733.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model_save'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("input:0")
    yhat = graph.get_tensor_by_name("Adam_1:0")
    print('Successfully load the pre-trained model!')

def predict():
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41 3).
        Test a single example.
        Arg:
                txtdata: Array in C.
        Returns:
            Three coordinates of a face normal.
    """
    image = tf.gfile.GFile('./rose.jpg','rb').read()   #加载原始图像
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (224, 224))
    # 标准化,使图片的均值为0，方差为1
    image = tf.image.per_image_standardization(image)
    #global X, yhat
    
    #data = np.array(txtdata)
    #data = data.reshape(-1, 41, 41, 41, 3)
    output = sess.run(yhat, feed_dict={X: image})  # (-1, 3)
    output = output.reshape(-1, 1)
    ret = output.tolist()
    return ret


load_model()
#testdata = np.fromfile('/home/senius/python/c_python/test/04t30t00.npy', dtype=np.float32)
#testdata = testdata.reshape(-1, 41, 41, 41, 3) # (150, 41, 41, 41, 3)
#testdata = testdata[0:2, ...] # the first two examples
#txtdata = testdata.tolist()
output = predict()
print(output)
#  [[-0.13345889747142792], [0.5858198404312134], [-0.7211828231811523], 
# [-0.03778800368309021], [0.9978875517845154], [0.06522832065820694]]