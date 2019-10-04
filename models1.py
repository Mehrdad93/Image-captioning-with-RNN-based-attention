import tensorflow as tf
from tensorflow.contrib import grid_rnn
class ICG_model:
    def __init__(self, features: dict, hyperparameters: dict, is_train: bool):
        batch_size = 40
        print(type(features))
        # Randomly collect instances into batches
        if is_train:
            #feat_batch, pb_batch, x_batch, y_batch = tf.train.shuffle_batch([features['feat'], features['x'],
             #                                        features['y']], batch_size=batch_size, capacity=20000, min_after_dequeue=200)
            feat_batch, x_batch, y_batch = tf.train.shuffle_batch([features['feat'], features['x'],
                                                                             features['y']], batch_size=batch_size,
                                                                            capacity=20000, min_after_dequeue=200)
            x_batch_dense = tf.sparse_tensor_to_dense(sp_input=x_batch, default_value=0, validate_indices=True,
                                                      name=None)
            y_batch_dense = tf.sparse_tensor_to_dense(sp_input=y_batch, default_value=0, validate_indices=True,
                                                      name=None)
        else:
            feat_batch = features['feat']
            x_batch_dense = features['x']
            y_batch_dense = features['y']
            x_batch = features['x']
            y_batch = features['y']


        vocab_size = hyperparameters['vocab_size']
        word_emb_size = hyperparameters['word_emb_size']
        if is_train:
            dropout_keep_rate = hyperparameters['dropout_keep_rate']
        else:
            dropout_keep_rate = 1.0

        emb = tf.Variable(tf.random_normal([vocab_size, word_emb_size]))
        x_batch_dense_emb = tf.nn.embedding_lookup(emb, x_batch_dense) # B x T x word_emb_size

        lstm_cap_layer1 = tf.contrib.rnn.BasicLSTMCell(word_emb_size, state_is_tuple=False)


        W_feat = tf.Variable(tf.random_normal([2048, word_emb_size]))
        b_feat = tf.Variable(tf.zeros([word_emb_size]))
        feat_proj = tf.tensordot(feat_batch, W_feat, [[1], [0]]) + b_feat
        feat_proj = tf.reshape(feat_proj, [-1, 1, word_emb_size])
        x_batch_dense_emb_feat = tf.concat([feat_proj, x_batch_dense_emb[:, 1:, :]], axis=1)  # B x T x word_emb_size
        #all_outputs_cap_layer1, _ = tf.nn.dynamic_rnn(lstm_cap_layer1, x_batch_dense_emb_pb, dtype=tf.float32) # B x T x word_emb_size
        all_outputs_cap_layer1, _ = tf.nn.dynamic_rnn(lstm_cap_layer1, x_batch_dense_emb_feat, dtype=tf.float32) # B x T x word_emb_size

        #print('**')
        #print(all_outputs_cap_layer1)
        self.__feat_batch = feat_batch # B x 2048

       

        W_predict = emb
        b_predict = tf.Variable(tf.zeros([vocab_size]))

        all_outputs_cap_layer1_dropout = tf.nn.dropout(all_outputs_cap_layer1, dropout_keep_rate)

        lgt = tf.tensordot(all_outputs_cap_layer1_dropout[:, 1:, :], W_predict, [[2], [1]]) + b_predict


        probs = tf.nn.softmax(lgt)
        #print('**')
        #print(lgt)
        #print(y_batch_dense)
        mask = tf.cast(y_batch_dense > 0, dtype=tf.float32)
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch_dense, logits=lgt)

        #print('**')
        #print(y_batch_dense)
        #print(mask)
        cost_mask = tf.multiply(mask, cost)
        cost_mask_sum = tf.reduce_sum(cost_mask, 1)
        cross_entropy = tf.reduce_mean(cost_mask_sum)
        learning_rate = hyperparameters['learning_rate']
        #optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9)
        train_step = optimizer.minimize(cross_entropy)
        info = [tf.shape(y_batch_dense), tf.shape(x_batch_dense_emb), tf.shape(mask), tf.shape(lgt), cost, cross_entropy]

        self.__train_step = train_step
        self.__info = info
        self.__cross_entropy = cross_entropy
        self.__W_predict = W_predict
        self.__x_batch = x_batch
        self.__y_batch = y_batch
        self.__probs = probs
        self.__y_batch_dense = y_batch_dense
        self.__mask = mask
        self.__lgt = lgt
        self.__cost = cost



    @property
    def _train_step(self):
        return (self.__train_step)

    @property
    def _info(self):
        return (self.__info)

    @property
    def _cross_entropy(self):
        return (self.__cross_entropy)

    @property
    def _W_predict(self):
        return (self.__W_predict)

    @property
    def _x_batch(self):
        return (self.__x_batch)

    @property
    def _y_batch(self):
        return (self.__y_batch)

    @property
    def _probs(self):
        return (self.__probs)

    @property
    def _feat_batch(self):
        return (self.__feat_batch)

    @property
    def _y_batch_dense(self):
        return (self.__y_batch_dense)

    @property
    def _mask(self):
        return (self.__mask)

    @property
    def _lgt(self):
        return (self.__lgt)

    @property
    def _cost(self):
        return (self.__cost)
