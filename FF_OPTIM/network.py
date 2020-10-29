import random

import numpy as np

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1 - sigmoid(z))


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
    error = self.d_cross_entropy(zs[-1], acts[-1], y)
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

  def d_cost(self, z, out, y):
    return (out - y) * sigmoid_prime(z)

  def d_cross_entropy(self, z, out, y):
    return (out - y)

