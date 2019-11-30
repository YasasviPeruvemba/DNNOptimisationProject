import random
import numpy as np
import pickle
import os
import gzip
import pandas as pd
import urllib.request
import scipy.sparse
import matplotlib.pyplot as plt
  
def estimate_coef(x, y): 
    n = np.size(x) 
  
    m_x, m_y = np.mean(x), np.mean(y) 
  
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    lambd = SS_xy / SS_xx 
    theta = m_y - lambd*m_x 
  
    return(theta, lambd) 
  
def plot_regression_line(x, y, b):  
    plt.scatter(x, y, color = "m", marker = "o", s = 30) 
  
    y_pred = b[0] + b[1]*x 
  
    plt.plot(x, y_pred, color = "g") 
  
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    plt.show() 
  
def main(): 
    pickle_in = open('RegressionDataset.pickle','rb')
    dataset = pickle.load(pickle_in)
    pickle_in.close()

    values = []
    final = []
    x=np.zeros((20))
    y=np.zeros((20))
    for i in range(5):
        for image in dataset[i]:
            k = 0
            for data in image:
                x[k] = data[0]
                y[k] = data[1]
                k = k + 1
            b = estimate_coef(y, x)
            values.append(b)

        lam = 0
        the = 0
        for value in values:
            the = the + value[0]
            lam = lam + value[1]

        lam = lam/len(values)
        the = the/len(values)
        final.append([lam,the])
    for i in range(5):
        print(final[i][0])
    print('##############')
    for i in range(5):
        print(final[i][1])
  
if __name__ == "__main__": 
    main()
