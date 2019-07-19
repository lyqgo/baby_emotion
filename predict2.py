import tensorflow as tf
import mobilenet_v2

checkpoint='./src/model.ckpt-7000'

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):     #change1
    logits, endpoints = mobilenet_v2.mobilenet(images,num_classes=3,is_training=False)
#logits,end_points = mobilenet_v2.mobilenet(images, num_classes=5, is_training=False)
  
# Restore using exponential moving average since it produces (1.5-2%) higher 
# accuracy
ema = tf.train.ExponentialMovingAverage(0.999)
vars = ema.variables_to_restore()

saver = tf.train.Saver(vars)


from IPython import display
import pylab
#from datasets import imagenet
import PIL
display.display(display.Image('./test_images/laugh1.jpg'))

with tf.Session() as sess:
  saver.restore(sess, checkpoint)
  #saver = tf.train.import_meta_graph('./model_save/model.ckpt-5733.meta')
  #saver.restore(sess, tf.train.latest_checkpoint('./model_save'))
  x = endpoints['Predictions'].eval(feed_dict={file_input: './test_images/laugh1.jpg'})
#label_map = imagenet.create_readable_names_for_imagenet_labels()  
#print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
  print("Top 1 prediction: ", x.argmax(), x.max())