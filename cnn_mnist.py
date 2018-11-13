from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle
import tensorflow as tf
import numpy as np


class CNN_model:


    def __init__(self, input, labels, x_valid, y_valid, x_test, y_test, lr, filter_size, num_filters=16, epochs=12, batch_size=32):
        self.epochs = epochs
        self.learning_rate = lr
        self.num_filters = num_filters
        self.batch_size = batch_size
        #self.x = input
        #self.y = labels
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.filter1 = None # filter for conv1
        self.filter2 = None # filter for conv2
        self.filter_size = filter_size

    def init_placeholders(self, x_shape, y_shape):
        #x = tf.placeholder(tf.float32, [None, 784])
        self.x = tf.placeholder(tf.float32, shape=[None,28,28,1])
        #self.x = tf.reshape(self.x, [-1,28,28,1])
        self.y = tf.placeholder(tf.float32, shape=[None,y_shape[1]])
        #print("x_shape", self.x.shape, "y_shape", self.y.shape)

    # couldnt simply give conv layers filters as a list, had to init seperately
    def init_filters(self):
        #self.filter1 = tf.get_variable('filter1', shape=[3,3,1,self.num_filters])
        #self.filter2 = tf.get_variable('filter2', shape=[3,3,self.num_filters,self.num_filters])
        self.filter1 = tf.get_variable('filter1', shape=[self.filter_size,self.filter_size,1,self.num_filters])
        self.filter2 = tf.get_variable('filter2', shape=[self.filter_size,self.filter_size,self.num_filters,self.num_filters])


    def run_layers(self, input):

        # first convolutional layer + relu + max pool
        conv1 = tf.layers.conv2d(input, filters=self.num_filters, kernel_size=[self.filter_size, self.filter_size], strides=(1, 1), padding='SAME')
        relu1 = tf.nn.relu(conv1)
        maxPool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

        # second convolutional layer + relu + max pool
        conv2 = tf.layers.conv2d(maxPool1, filters=self.num_filters, kernel_size=[self.filter_size, self.filter_size], strides=(1, 1), padding='SAME')
        relu2 = tf.nn.relu(conv2)
        maxPool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

        # first fully connected layer
        fc1 = tf.contrib.layers.fully_connected(maxPool2, 128, activation_fn=tf.nn.relu)
        flatten = tf.contrib.layers.flatten(fc1)
        # second fully connected layer (flattened bc output)
        output = tf.contrib.layers.fully_connected(flatten, 10, activation_fn=None)

        return output


    def train(self, x_train, y_train, x_valid, y_valid, x_test, y_test):
        dev = '/GPU:0'
        tf.device(dev)
        tf.reset_default_graph()
        #print("x_train:", x_train.shape, "y_train", y_train.shape)

        self.init_placeholders(x_train.shape, y_train.shape)
        self.init_filters()

        output = self.run_layers(self.x)
        y_pred = tf.argmax(output ,1)

        train_cost = np.zeros((self.epochs))
        train_accuracy = np.zeros((self.epochs))
        valid_accuracy =    np.zeros((self.epochs))

        # init loss and optimizer
        cross_entropy  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.y))
        #cross_entropy = self.cost(y_pred)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)

        # setup variable initialization
        init_op = tf.global_variables_initializer()


        correct_prediction = tf.equal(tf.argmax(self.y, 1), y_pred)
        # accuracy =  (correct values / all values) (boolean casted as 0/1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #error = 1 - accuracy

        print("train_cost", train_cost.shape)
        with tf.Session() as sess:
            # actual batchwise training begins
            sess.run(init_op)
            num_batches_train = int(len(y_train) // self.batch_size)
            num_batches_valid = int(len(y_valid) // self.batch_size)
            #num_batches = x_train.shape[0] // self.batch_size
            print("num_batches", num_batches_train, num_batches_valid)
            for epoch in range(self.epochs):
                valid_avg_acc = 0
                train_avg_acc = 0
                valid_acc = 0
                train_acc = 0
                cost = 0
                for b in range(num_batches_train):
                    x_batch_train = x_train[b*self.batch_size:((b+1)*self.batch_size)]
                    y_batch_train = y_train[b*self.batch_size:((b+1)*self.batch_size)]
                    _, cost = sess.run([optimizer, cross_entropy], feed_dict={self.x: x_batch_train, self.y: y_batch_train})
                    train_cost[epoch] += cost / num_batches_train
                    train_accuracy[epoch] += accuracy.eval(feed_dict={self.x: x_batch_train, self.y: y_batch_train}) / num_batches_train
                    #train_acc = sess.run(accuracy, feed_dict={self.x: x_batch_train, self.y: y_batch_train})
                    #train_avg_acc += train_acc / num_batches_train
                    #train_accuracy[]
                    #train_accuracy[epoch] = accuracy.eval({self.x: x_batch_train, self.y: y_batch_train})

                for i in range(num_batches_valid):
                    x_batch_valid = x_valid[i*self.batch_size:((i+1)*self.batch_size)]
                    y_batch_valid = y_valid[i*self.batch_size:((i+1)*self.batch_size)]
                    valid_accuracy[epoch] += accuracy.eval(feed_dict={self.x: x_batch_valid, self.y: y_batch_valid}) / num_batches_valid
                    #valid_acc = sess.run(accuracy, feed_dict={self.x: x_batch_valid, self.y:y_batch_valid})
                    #valid_avg_acc += valid_acc /
                print("Epoch:", epoch, "cost:", "{:.3f}".format(train_cost[epoch]), "train_acc: ", "{:.3f}".format(train_accuracy[epoch]), "valid_acc", "{:.3f}".format(valid_accuracy[epoch]))


            num_batches_test = int(len(y_test)/self.batch_size)
            avg_error = 0
            for k in range(num_batches_test):
                x_batch_test = x_test[(k*self.batch_size):((k+1)*self.batch_size)]
                y_batch_test = y_test[(k*self.batch_size): ((k+1)*self.batch_size)]
                test_accuracy = sess.run(accuracy, feed_dict = {self.x: x_batch_test, self.y: y_batch_test})
                avg_error += test_accuracy / num_batches_test
            #print("test_error", "{:.3f}".format(avg_error))
            print("test_error", avg_error)

                #saver = tf.train.Saver()
                #saver.save(sess, 'model')

        """
        def test(x_test, y_test):
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph('my_test_model-1000.meta')
                saver.restore(sess,tf.train.latest_checkpoint('./'))
                graph = tf.get_default_graph()
        """

        return train_cost, train_accuracy, valid_accuracy.tolist()




def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def train_and_validate(x_train, y_train, x_valid, y_valid, x_test, y_test, lr=0.01, filter_size=7, num_filters=16, epochs=12, batch_size=32):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    model = CNN_model(x_train, y_train, x_valid, y_valid, lr, filter_size, num_filters, epochs, batch_size)
    train_cost, train_accuracy, learning_curve = model.train(x_train, y_train, x_valid, y_valid, x_test, y_test)

    #return learning_curve, model  # TODO: Return the validation error after each epoch (i.e learning curve) and your model
    return learning_curve, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs

    #### train and test convolutional neural network ####

    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)
    #x_train, y_train, x_valid, y_valid, x_test, y_test = mnist()

    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, x_test, y_test)
    #cost  = train_and_validate(x_train, y_train, x_valid, y_valid)

    #test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them Cinto a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve
    #results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
