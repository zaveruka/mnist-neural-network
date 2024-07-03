import numpy as np
import pandas as pd
import matplotlib as plt

data = pd.read_csv("/home/codetry-6/Documents/projects/zaveruka/neural-network-py/train.csv") # Path to training file
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1: n]

def init_params():
  W1 = np.random.rand(10, 784)
  b1 = np.random.rand(10, 1)
  W2 = np.random.rand(10, 10)
  b2 = np.random.rand(10, 1)
  return W1, b1, W2, b2

def ReLU(Z):
  return p.maximum(0, Z)

def ReLU_deriv(Z):
    return Z > 0

def softmax(Z):
  return np.exp(Z) / np.sum(np.exp(Z))

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arrange(Y.size, Y )] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y
