from os import listdir
from os.path import isfile, join
from scipy.io import loadmat
import tensorflow as tf

_data_type = 'train'    # train, val, test
_year = '2014'  # 2014, 2015
_layer = '482'  # 480, 482, 484, ...
_path = './Desktop/prj/res-feat/layer' + _layer + '/' + _data_type + _year + '/'

# get an input for Multi Layer Perceptron by giving a loaded mat file of features
def get_mlp_input(x):
    feat = x['feat' + _layer]
    _d1 = x['feat' + _layer].shape[0]
    _d2 = x['feat' + _layer].shape[1]
    _d3 = x['feat' + _layer].shape[2]

    R = []  # will be _d1 *_d2 vector of vectors of length _d3
    for i in range(_d1):
        for j in range(_d2):
            v_temp = []  # a _d3 length vector
            for k in range(_d3):
                v_temp.append(feat[i, j, k])
            R.append(v_temp)
    return tf.convert_to_tensor(R, dtype=tf.float32), _d1, _d2, _d3

#------------ IMPLEMENTING THE MODEL --------------

# Network Parameters
n_lstm_ht_size = 100
n_Ri_size = _d3
n_input = n_lstm_ht_size + n_Ri_size    # (|h_t - 1| + |R_i|)
n_hidden_1 = 50  # 1st layer number of features
n_classes = 1  # the output

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden layer with ReLU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


#-------------------------------------------
files = [f for f in listdir(_path) if isfile(join(_path, f))]
for file in files:
    loaded_mat = loadmat(_path + file)

    # R is a _d1 *_d2 vector of vectors of length _d3
    R, _d1, _d2, _d3 = get_mlp_input(loaded_mat)



    #----------------------- USING THE MODEL --------------------------------

    # Parameters
    learning_rate = 0.001
    training_epochs = 1
    batch_size = _d1 * _d2  # equals to |R|
    display_step = 1

    # Construct model
    logits = multilayer_perceptron(x)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            #batch_x, batch_y = mnist.py.train.next_batch(batch_size)
            batch_x = R
            batch_y = R
            # Run optimization op (backprop) and cost op (to get loss value)
            _, _loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(_loss))
        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print("Accuracy:", accuracy.eval({X: mnist.py.test.images, Y: mnist.py.test.labels}))
