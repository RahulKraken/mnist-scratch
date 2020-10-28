#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dataset

import pickle
import gzip
import random

import numpy as np

def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    train_data, valid_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (train_data, valid_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# # Documentation
# 
# ### Constructor
# 
# Takes in sizes of layers
# for example [5, 10, 2] means input layer of size 5, one hidden layer of size 10 and output layer of size 2
# and [10, 20, 20, 30, 5] means input layer of size 10, 3 hidden layers of size 20, 20 and 30 and an output layer of size 5

# In[17]:


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# In[18]:


# Neural network

class Network():
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, batch_size, epochs, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        # we need to take small batch of data in each epoch and train on that
        for it in range(epochs):
            random.shuffle(training_data)
            batch = [training_data[x : x + batch_size] for x in range(0, n, batch_size)]
            for mini_batch in batch:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}.".format(it, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} completed.".format(it))
    
    def update_mini_batch(self, mini_batch, eta):
        # total change in network for this mini_batch
        total_dw = [np.zeros(w.shape) for w in self.weights]
        total_db = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            # dw, db = change in weights and biases for this particular x, y...??
            dw, db = self.train(x, y)
            # it's just like sum += d[i] => sum = sum + d[i]
            # total_dw = total_dw + dw but in complicated vector way
            total_dw = [nw + dnw for nw, dnw in zip(total_dw, dw)]
            total_db = [nb + dnb for nb, dnb in zip(total_db, db)]
        # update network for this mini_batch
        self.weights = [w - (eta / len(mini_batch)) * dnw for w, dnw in zip(self.weights, total_dw)]
        self.biases = [b - (eta / len(mini_batch)) * dnb for b, dnb in zip(self.biases, total_db)]
    
    def train(self, x, y):
        # backpropagration starts at the end propagates backwards
        # find output error
        # for that we need to feed forward some x and see the output
        # find the difference from the actual output
        
        # store total impact of this particular x, y on the network
        total_db = [np.zeros(b.shape) for b in self.biases]
        total_dw = [np.zeros(w.shape) for w in self.weights]
        
        # we need z and a vector <3rd and 4th eq of backpropagation>
        act = x
        acts = [x]
        zs = []
        
        # first feed forward and see how it deals with the input
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, act) + b
            zs.append(z)
            act = sigmoid(z)
            acts.append(act)
        
        # now back propagation with error of last layer
        error = self.d_cost(acts[-1], y) * sigmoid_prime(zs[-1])
        total_db[-1] = error
        total_dw[-1] = np.dot(error, acts[-2].transpose())
        # now move backwards and update each layers weights and biases
        for l in range(2, self.num_layers):
            sp = sigmoid_prime(zs[-l])
            error = np.dot(self.weights[-l + 1].transpose(), error) * sp
            total_db[-l] = error
            total_dw[-l] = np.dot(error, acts[-l - 1].transpose())
        return (total_dw, total_db)
    
    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)
    
    def d_cost(self, out, y):
        return (out - y)


# In[19]:


train, valid, test = load_data_wrapper()
net = Network([784, 30, 50, 10])
net.SGD(train, 10, 30, 3.0, test)

