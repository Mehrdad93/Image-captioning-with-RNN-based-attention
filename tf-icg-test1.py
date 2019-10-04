import tensorflow as tf
import os
import numpy
from models1 import ICG_model
import scipy.io
import pickle
import json
f = open('/home/hamid/Desktop/prj/dictionaries/newdict.pickle','rb')
newdict = pickle.load(f)
f.close()

class refnode:

    def __init__(self, obj = None):
           if obj != None:
               obj = None
           self.obj = obj
class node:
    def __init__(self, sen = None, prob = None, fin = None):
        if sen != None:
            sen = None
        if prob != None:
            prob = None
        if fin != None:
            fin = None
        self.sen = sen
        self.prob = prob
        self.fin = fin


def beam_search(feat_batch, feat_batch_val, x_batch, maxlen):
    beam_size = 20
    x_batch_val = [[0, 0]]
    goodnodes = []
    goodprobs = []
    for i in range(maxlen):
        if (i == 0):

            q = sess.run(probs, feed_dict = {feat_batch: feat_batch_val, x_batch:  x_batch_val})
            
            temp = q[0, 0, :].argsort()
            temp = temp[::-1][:beam_size]
            temp = temp[:beam_size]
            for k in range(beam_size):
                refnd = refnode()
                nd = node()
                nd.sen = [[0, 0]]
                nd.sen[0].append(temp[k])
                nd.fin = 0
                if (temp[k] == 1):
                    nd.fin = 1
                nd.prob = q[0, 0, temp[k]]
                refnd.obj = nd
                goodnodes.append(refnd)
                goodprobs.append(nd.prob)
        else:
            allnodes = []
            allprobs = []
            for k in goodnodes:
                if (k.obj.fin == 0):
                    x0 = k.obj.sen
                    x_batch_val = x0
                    q = sess.run(probs, feed_dict={feat_batch: feat_batch_val, x_batch: x_batch_val})

                    temp = q[0, i, :].argsort()
                    temp = temp[::-1][:beam_size]
                    temp = temp[:beam_size]

                    for j in range(beam_size):

                        refnd = refnode()
                        nd = node()
                        nd.sen = [[]]
                        for t in k.obj.sen[0]:
                            nd.sen[0].append(t)
                        nd.sen[0].append(temp[j])
                        nd.fin = 0
                        if (temp[j] == 1):
                            nd.fin = 1
                        nd.prob = q[0, i, temp[j]]
                        nd.prob = nd.prob * k.obj.prob
                        refnd.obj = nd
                        allnodes.append(refnd)
                        allprobs.append(nd.prob)
                else:
                    allnodes.append(k)
                    allprobs.append(k.obj.prob)
            indxs = sorted(range(len(allprobs)), key=lambda j: allprobs[j])
            r = len(indxs) - beam_size
            indxs = indxs[r:]
            indxs = indxs[::-1]
            goodnodes = []
            goodprobs = []
            for k in indxs:
                goodnodes.append(allnodes[k])
                goodprobs.append(allprobs[k])
        cc = 0
        for k in goodnodes:
            if (k.obj.fin == 1):
                cc = cc + 1
        if (cc == beam_size):
            break
    temp = goodnodes[0]
    x0 = temp.obj.sen

    xx0 = [x0[0][:-1]]


    # converting to words
    # sentence = []
    sentence = ''
    for i in x0[0]:
        for key, value in newdict.items():
            if value == i:
                # sentence.append(key)
                sentence = sentence + key + ' '
                break
    #return sentence[:-1] + ' .', xx0
    return sentence[:-1] , xx0

#DIR = "/home/hamid/Desktop/prj/icg"
DIR = "/home/hamid/Documents/save/save"
hyperparameters = {'batch_size': 1, 'save_freq': 100, 'vocab_size': len(newdict) + 2, 'word_emb_size': 512, 'dropout_keep_rate': 0.6, 'learning_rate': 0.0001}

features = {'feat': tf.placeholder(tf.float32, [None, 2048]),
    'x':  tf.placeholder(tf.int32, [None, None]), 'y': tf.placeholder(tf.int32, [None, None])}


model = ICG_model(features, hyperparameters, is_train=False)
train_step = model._train_step
info = model._info
cross_entropy = model._cross_entropy
W_predict = model._W_predict
x_batch = model._x_batch
y_batch = model._y_batch
probs = model._probs
feat_batch = model._feat_batch

maxlen = 20
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(W_predict.eval())
    saver.restore(sess, tf.train.latest_checkpoint(DIR))
    #saver.restore(sess, os.path.join(DIR,"model-19700"))
    print(W_predict.eval())

    '''
    f = open('/cs/oschulte/mkhademiatoschulte/Theano-0.6.0/coco_val2014_x.pickle','rb')
    coco_test2014_x = pickle.load(f)
    f.close()
    f = open('/cs/oschulte/mkhademiatoschulte/Theano-0.6.0/coco_val2014_y.pickle','rb')
    coco_test2014_y = pickle.load(f)
    f.close()
    f = open('/cs/oschulte/mkhademiatoschulte/Theano-0.6.0/imageids_test2014.pickle','rb')
    imageids_test2014 = pickle.load(f)
    f.close()
    test_set = (coco_test2014_x, coco_test2014_y, imageids_test2014)
    '''

    imageids_test2014 = []
    with open('/home/hamid/Desktop/prj/step1output/test_images_514.txt') as json_file:
        data = json.load(json_file)
        for p in data['images']:
            # imageids_test2014.append(p['filename'])
            imageids_test2014.append(int(p['filename'][4:-4]))


    '''
    dir_train2014 = os.listdir('/home/hamid/Desktop/prj/MSCOCO/images/test2014/')
    imageids_test2014 = []
    for img_name in dir_train2014[:2000]:
        imageids_test2014.append(int(img_name[14:-4]))
    '''


    test_set = (None, None, imageids_test2014)

    maxlen = 20
    all_captions = []
    all_imids = []
    print('the size: ', len(test_set[2]))
    for i in range(len(test_set[2])):
        #print(i, test_set[2][i])
        temp_dict = {}
        imid = test_set[2][i]

        if (imid not in all_imids):
            temp_dict['image_id'] = int(imid)

            #feat = scipy.io.loadmat('/home/hamid/Desktop/prj/res-feat/layer514/test2014/res_' + str(imid) + '.mat')
            feat = scipy.io.loadmat('/home/hamid/Desktop/prj/res-feat/layer514/val2014/res_' + str(imid) + '.mat')
            feat = feat['feat514'].astype(float)
            feat = numpy.reshape(feat, (1,-1), order = 'C')
            #print(feat.shape)
            sentence, xx0 = beam_search(feat_batch, feat, x_batch, maxlen)
            print(imid, sentence)
            print(xx0)

            temp_dict['caption'] = sentence
            all_captions.append(temp_dict)
            all_imids.append(imid)
    import json

    with open('/home/hamid/Desktop/prj/results/captions_val2014_fakecap_results_dropout_5000.json','w') as outfile:
        json.dump(all_captions, outfile)
    outfile.close()


