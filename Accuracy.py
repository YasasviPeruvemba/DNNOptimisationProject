import random
import numpy as np
import pickle
import os
import gzip
import pandas as pd
import urllib.request
import scipy.sparse
import matplotlib.pyplot as plt

class OptiNetwork(object):

    def __init__(self, sizes,bits,weights=None):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.bits = bits

        if weights:
            self.weights = weights
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a, flag):
        if flag==1:    
            for i in range(5):
                #print(len(a))
                #print(a.shape)
                for j in range(len(a)):
                    val = int(a[j]*2**self.bits[i])
                    val = val/2**self.bits[i]
                    a[j] = np.float32(val)
                a = sigmoid(np.dot(self.weights[i], a)+self.biases[i])
            return a
        else:
            for i in range(5):
                a = sigmoid(np.dot(self.weights[i], a)+self.biases[i])
            return a

    def Accuracy(self, test_data, epochs):
        test_data = list(test_data)
        n_test = len(test_data)
        for i in range(epochs):
            print("Accuracy without Bit Assignment of Epoch {0} : {1} / {2}".format(i+1,self.evaluate(test_data), n_test))
            print("Accuracy with Bit Assignment of Epoch {0} : {1} / {2}".format(i+1,self.evaluateNew(test_data), n_test))
                
    def evaluateNew(self, test_data):
        test_results = [(np.argmax(self.feedforward(x,1)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x,0)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

pickle_in = open('DigitRecognizerWeights.pickle','rb')
wts = pickle.load(pickle_in)
pickle_in.close()

net = OptiNetwork([784,160,80,40,20,10],[2,3,5,6,6],wts)

import mnist_loader
train_data,valid_data,test_data = mnist_loader.load_data_wrapper()

train_data = list(train_data)
test_data = list(test_data)
net.Accuracy(test_data,1)

