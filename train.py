
#coding:utf-8

import os
import numpy as np
import tensorflow as tf
import glob

import model

tf.reset_default_graph()
def get_files(file_dir):
    image_list, label_list = [], []
    for label in os.listdir(file_dir):
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
with open("fer_label.txt", 'r') as f:
    for line in f.readlines():
        folder, label = line.strip().split(':')[0], line.strip().split(':')[1]
        label_dict[folder] = label
        label_dict_res[label] = folder
print(label_dict)


train_dir = "./datasets/five_train"
logs_train_dir = './model_save1'
init_lr = 0.1
BATCH_SIZE = 64
train, train_label = get_files(train_dir)
one_epoch_step = len(train) / BATCH_SIZE
decay_steps = 20*one_epoch_step
MAX_STEP = int(100*one_epoch_step)
N_CLASSES = len(label_dict)
IMG_W = 224
IMG_H = 224
CAPACITY = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # gpu编号
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
    # 随机左右翻转
    image = tf.image.random_flip_left_right(image)
    # 随机上下翻转
    image = tf.image.random_flip_up_down(image)
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=32/255.0)
    # 随机设置图片的对比度
    #image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # 随机设置图片的色度
    image = tf.image.random_hue(image, max_delta=0.05)
    # 随机设置图片的饱和度
    #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # 标准化,使图片的均值为0，方差为1
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)

    tf.summary.image("input_img", image_batch, max_outputs=5)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

def main():    
    global_step = tf.Variable(0, name='global_step', trainable=False)    
    # label without one-hot
    batch_train, batch_labels = get_batch(train,
                                          train_label,
                                          IMG_W,
                                          IMG_H,
                                          BATCH_SIZE, 
                                          CAPACITY)
    # network
    logits = model.MobileNetV2(batch_train, num_classes=N_CLASSES, is_training=True).output
    #logits = model.model2(batch_train, BATCH_SIZE, N_CLASSES)
    #logits = model.model4(batch_train, N_CLASSES, is_trian=True)
    print(logits.get_shape())
    # loss

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.summary.scalar('train_loss', loss)
    # optimizer

    lr = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step, decay_steps=decay_steps, decay_rate=0.1)
    tf.summary.scalar('learning_rate', lr)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

    # accuracy
    correct = tf.nn.in_top_k(logits, batch_labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('train_acc', accuracy)

    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=config)
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    #saver = tf.train.Saver()
    var_list = tf.trainable_variables() 
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver.restore(sess, logs_train_dir+'/model.ckpt-174000') 

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                    break
            _, learning_rate, tra_loss, tra_acc = sess.run([optimizer, lr, loss, accuracy])
            if step % 100 == 0:
                print('Epoch %3d/%d, Step %6d/%d, lr %f, train loss = %.2f, train accuracy = %.2f%%' %(step/one_epoch_step, MAX_STEP/one_epoch_step, step, MAX_STEP, learning_rate, tra_loss, tra_acc*100.0))

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                
                variable_names = [v.name for v in tf.trainable_variables()[-6:-1]]
                values = sess.run(variable_names)
                for k,v in zip(variable_names, values):
                    mean=np.mean(v)
                    std=np.std(v)
                    #mean, variance = tf.nn.moments(v, axes=[0,1])
                    print("Variable: ", k)
                    print("Shape: ", v.shape)
                    print('mean:',mean)
                    print('var:',std)



            if step % 1000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()


    coord.join(threads)
    sess.close()
 

if __name__ == '__main__':
    main()
