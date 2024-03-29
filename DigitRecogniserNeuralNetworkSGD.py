import random
import numpy as np

class Network(object):

    def __init__(self, sizes,weights=None):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        if weights:
            self.weights = weights
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)

        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        training_data = list(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j+1))

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):

        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = []
        activations.append(x)
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for q in range(2, self.num_layers):
            z = zs[-q]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-q+1].transpose(), delta) * sp
            delta_nabla_b[-q] = delta
            delta_nabla_w[-q] = np.dot(delta, activations[-q-1].transpose())

        return delta_nabla_b, delta_nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    return sigmoid(z)*(1-sigmoid(z))


# Actual recognition of digits
import pickle
import os
import gzip
import pandas as pd
import urllib.request
import scipy.sparse
import matplotlib.pyplot as plt

def load_dataset():
    
    def download(filename,source='http://yann.lecun.com/exdb/mnist/'):

        print ('Downloading ',filename)
        urllib.request.urlretrieve(source+filename,filename)

    def load_images(filename):
        
        if not os.path.exists(filename):
            download(filename)

        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = np.reshape(data,(-1,784,1))
        
        return data/np.float32(256)

    def load_labels(filename):

        if not os.path.exists(filename):
            download(filename)
        
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    X_Train = load_images('train-images-idx3-ubyte.gz')
    Y_Train = load_labels('train-labels-idx1-ubyte.gz')
    X_Test = load_images('t10k-images-idx3-ubyte.gz')
    Y_Test = load_labels('t10k-labels-idx1-ubyte.gz')

    return X_Train, Y_Train, X_Test, Y_Test

X_Train, Y_Train, X_Test, Y_Test = load_dataset()

import mnist_loader
train_data,valid_data,test_data = mnist_loader.load_data_wrapper()

X_data = []

for x,y in zip(X_Train,Y_Train):
    keypair = []
    keypair.append(x)
    keypair.append(y)
    X_data.append(keypair)

Y_data = []

for x,y in zip(X_Test,Y_Test):
    keypair = []
    keypair.append(x)
    keypair.append(y)
    Y_data.append(keypair)

pickle_in = open('DigitRecognizerWeightsSGD.pickle','rb')
wts = pickle.load(pickle_in)
pickle_in.close()

# Example of image
plt.imshow(scipy.reshape(X_Train[2],[28,28]))
plt.show()

train_data = list(train_data)
test_data = list(test_data)

net = Network([784,160,80,40,20,10])
net.SGD(train_data,10,20,2.9,test_data)

# Saving weights for the future
# w = net.weights
# pickle_out = open('DigitRecognizerWeights.pickle','wb')
# pickle.dump(w,pickle_out)
# pickle_out.close()
