import random
import numpy as np
import pickle
import os
import gzip
import pandas as pd
import urllib.request
import scipy.sparse
import matplotlib.pyplot as plt
import Regression
########################################################

class OptiNetwork(object):

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

    def injectAndExtract(self, layer, images, y_output):
        regFinalInp = []
        for newimage,y in zip(images,y_output):
            regInp = []
            for i in range(20):
                image = newimage
                # guess delX
                delX = random.random()/10000;
                for j in range(layer):
                    #print(j,image.shape,self.weights[j].shape)
                    image = sigmoid(np.dot(self.weights[j], image)+self.biases[j])

                for i in range(len(image)):
                    err = random.uniform(-delX,delX)
                    image[i] = image[i] + err
                    
                for j in range(layer,5):
                    image = sigmoid(np.dot(self.weights[j], image)+self.biases[j])
                    #print(j,self.weights[j].shape,image.shape)

                #STD of (image-y)
                out_std = np.std(image-y)
                #Inp to regression
                regInp.append([delX,out_std])
            regFinalInp.append(regInp)
        return regFinalInp

########################################################

def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    return sigmoid(z)*(1-sigmoid(z))

########################################################

import mnist_loader
train_data,valid_data,test_data = mnist_loader.load_data_wrapper()

train_data = list(train_data)
# Getting the list of 50 images
images = []
for i in range(50):
    images.append(train_data[i][0])

print(images[0].shape)

pickle_in = open('DigitRecognizerWeights.pickle','rb')
wts = pickle.load(pickle_in)
pickle_in.close()

net = OptiNetwork([784,160,80,40,20,10],wts)

pickle_in = open('Y_Outputs.pickle','rb')
y_output = pickle.load(pickle_in)
pickle_in.close()

#Storing the outputs of the 50 images
#y_output = []
#for image in images:
  #  y_output.append(net.feedforward(image))
  #  print(y_output[-1])

# pickle_out = open('Y_Outputs.pickle','wb')
# pickle.dump(y_output,pickle_out)
# pickle_out.close()


dataset = []
for i in range(5):
    dataset.append(net.injectAndExtract(i,images,y_output))

print(len(dataset))
print(len(dataset[0]))

pickle_out = open('RegressionDataset.pickle','wb')
pickle.dump(dataset,pickle_out)
pickle_out.close()

Regression.main()


