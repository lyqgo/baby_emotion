# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:43:57 2019

@author: Administrator
"""


#coding:utf-8

import os, sys
import numpy as np
import tensorflow as tf
import glob
import tensorflow.contrib.slim as slim

# import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
# from tensorflow.contrib.slim.python.slim.nets import resnet_v2

sys.path.append("./src/slim")
import mobilenet_v2
 

def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
        if os.path.isfile(os.path.join(file_dir, label)):
            continue

        for img in glob.glob(os.path.join(file_dir, label, "*.jpg")):
            image_list.append(img)
            label_list.append(int(label_dict[label]))
    print('There are %d data' %(len(image_list)))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

label_dict, label_dict_res = {}, {}
# 手动指定一个从类别到label的映射关系

with open("./src/label/adult_label.txt", 'r') as f:   ######baby3_label
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)
 

finetune_model = './src/model.ckpt-7000'    #finetune_model = './model_save2/model.ckpt-100'
checkpoint_not_load_scope = 'MobilenetV2/Logits'  # not load fc layer
trainable_scope = 'MobilenetV2/Logits'  # train fc layer when finetune 

train_dir = "./datasets/three_train"  #######
logs_train_dir = './model_save3'    ##########
init_lr = 0.0001
weight_decay = 0.0001
BATCH_SIZE = 16
freeze_basemodel = True #True
train, train_label = get_files(train_dir)
one_epoch_step = len(train) / BATCH_SIZE
decay_steps = int(30*one_epoch_step)
MAX_STEP = int(100*one_epoch_step)
N_CLASSES = len(label_dict)

IMG_W = 224
IMG_H = 224
CAPACITY = 1000 + 3 * BATCH_SIZE
display_step = 100
batch_norm_params = {

        # Decay for the moving averages.
        'decay': 0.997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # gpu编号
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # 设置最小gpu使用量
 

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 数据增强

    #image = tf.image.resize_image_with_pad(image, target_height=image_W, target_width=image_H)
    image = tf.image.resize_images(image, (image_W, image_H))
    # random rotate 90
    if np.random.randn() > 0:
        image = tf.image.transpose_image(image)
    # 随机左右翻转
    image = tf.image.random_flip_left_right(image)
    # 随机上下翻转
    image = tf.image.random_flip_up_down(image)
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=32/255.0)
    # 随机设置图片的对比度
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 随机设置图片的色度
    #image = tf.image.random_hue(image, max_delta=0.05)
    # 随机设置图片的饱和度
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # 标准化,使图片的均值为0，方差为1
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size=batch_size,
                                                num_threads=64,
                                                capacity=capacity)

    tf.summary.image("input_img", image_batch, max_outputs=5)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

 
def get_finetuned_variables():
    exclusions = [scope.strip() for scope in checkpoint_not_load_scope.split(',')]
    variables_to_restore = []
 
    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除。
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # print("restore variables {}".format(variables_to_restore))
    return variables_to_restore

# 获取所有需要训练的变量列表。

def get_trainable_variables():
    scopes = [scope.strip() for scope in trainable_scope.split(',')]
    variables_to_trian = []

    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有需要训练的参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_trian.extend(variables)
    return variables_to_trian
 

def main():
    tf.reset_default_graph()
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # label without one-hot
    batch_train, batch_labels = get_batch(train,
                                          train_label,
                                          IMG_W,
                                          IMG_H,
                                          BATCH_SIZE, 
                                          CAPACITY)

    # network, set is_training=False when predict img
    # with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
    #     # logits, _ = inception_v3.inception_v3(batch_train, num_classes=N_CLASSES, is_training=True)
    #     logits, _ = resnet_v2.resnet_v2_152(batch_train, num_classes=N_CLASSES, is_training=True)
    #     logits = tf.reshape(logits, [-1, N_CLASSES])

    with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, weights_regularizer=slim.l2_regularizer(weight_decay)):

    # with slim.arg_scope(mobilenet_v2.training_scope()):
        logits, _ = mobilenet_v2.mobilenet(batch_train, num_classes=N_CLASSES, is_training=True)
    print(logits.get_shape())

    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    regularization_losses_n = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([loss] + regularization_losses_n, name='total_loss')

    tf.summary.scalar('train_loss', loss)

    # optimizer
    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=decay_steps, decay_rate=0.1)
    tf.summary.scalar('learning_rate', lr)

    # set optimizer, trainable variable

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if freeze_basemodel:
            trainable_variable = get_trainable_variables()
            for var in trainable_variable:
                print("only train variable:", var)

            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step, var_list=trainable_variable)
        else:
            print("train all variable")
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)  #train all var

 

    # accuracy
    correct = tf.nn.in_top_k(logits, batch_labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('train_acc', accuracy)

 
    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    # load model

    load_finetune_model = slim.assign_from_checkpoint_fn(finetune_model, get_finetuned_variables(),
                                                         ignore_missing_vars=True)

    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver.restore(sess, logs_train_dir+'/model.ckpt-174000')

    print('Loading finetune model from %s' % finetune_model)
    load_finetune_model(sess)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                    break

            _, learning_rate, tra_loss, tra_acc = sess.run([optimizer, lr, loss, accuracy])
            if step % display_step == 0:
                print('Epoch:%3d/%d, Step:%6d/%d, lr:%f, train loss:%.4f, train acc:%.2f%%' %(step/one_epoch_step+1, MAX_STEP/one_epoch_step, step+display_step, MAX_STEP, learning_rate, tra_loss, tra_acc*100.0))

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 500 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print("save model", checkpoint_path)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
 

if __name__ == '__main__':
    main()
