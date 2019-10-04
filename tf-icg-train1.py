from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib import grid_rnn
import scipy.io
import numpy
import pickle
#f = open('./Desktop/prj/dictionaries/newdict.pickle','rb')
f = open('./newdict.pickle','rb')
newdict = pickle.load(f)
f.close()
from models1 import ICG_model
rng = numpy.random.RandomState(1234)
#save_dir = "./Desktop/prj/step1output"
save_dir = "./"
#filename = os.path.join(save_dir, "train_val_482.tfrecords")     # grid_size is 7, feat_dim is 2048, batch_size = 100
#filename = os.path.join(save_dir, "train_val_480.tfrecords")  # grid_size is 14, feat_dim is 1024, batch_size = 40 (due to OOM)
#filename = os.path.join(save_dir, "train_val_514.tfrecords")  # grid_size is 1, feat_dim is 2048, batch_size = 40 (due to OOM)
filename = os.path.join(save_dir, "test_train_val_514.tfrecords")  # grid_size is 1, feat_dim is 2048, batch_size = 40 (due to OOM)
feat_dim = 2048
filename_queue = tf.train.string_input_producer([filename], num_epochs=20)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

hyperparameters = {'batch_size': 40, 'save_freq': 100, 'vocab_size': len(newdict) + 2, 'word_emb_size': 512, 'dropout_keep_rate': 0.6, 'learning_rate': 0.0001}



features = tf.parse_single_example(
    serialized_example,
    features={
        'feat': tf.FixedLenFeature([2048], tf.float32),
        'x': tf.VarLenFeature(tf.int64),
        'y': tf.VarLenFeature(tf.int64)
    })




print(type(hyperparameters))
model = ICG_model(features, hyperparameters, is_train=True)
train_step = model._train_step
info = model._info
x_batch = model._x_batch
y_batch = model._y_batch
probs = model._probs
feat_batch = model._feat_batch
cross_entropy = model._cross_entropy
y_batch_dense = model._y_batch_dense
mask = model._mask
lgt = model._lgt

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
init = tf.local_variables_initializer()
sess.run(init)

#load pre-trained:
#DIR = "/local-scratch/tf/model/icg1"
#saver = tf.train.Saver()
#saver.restore(sess, tf.train.latest_checkpoint(DIR))



# Coordinator
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)
#DIR = "/home/hamid/Desktop/prj/icg11"
DIR = "./save"
NUM_EPOCHS = 20
try:
    step = 0
    total_cost = 0.0
    while not coord.should_stop():
        step += 1

        info1 = sess.run(info)
        #print('**')
        #print(info1[0])
        #print(info1[2])
        #print(info1[3])
        print(info1[5])
        cross_entropy_cost = sess.run(cross_entropy)
        #print('**')
        sess.run(train_step)
        #print(info1)
        total_cost += cross_entropy_cost

        if step % hyperparameters['save_freq'] == 0:
            print('%d steps.' % (step))
            print(total_cost / hyperparameters['save_freq'])
            total_cost = 0.0
            saver.save(sess, os.path.join(DIR, "model"), global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
    # When done, ask the threads to stop
    coord.request_stop()
# Wait for threads to finish
coord.join(threads)
sess.close()

