import numpy as np
import pandas as pd
import matplotlib as plt
import os

current_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(current_dir, 'train.csv'))
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

def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1)
  return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1    
  W2 = W2 - alpha * dW2  
  b2 = b2 - alpha * db2    
  return W1, b1, W2, b2