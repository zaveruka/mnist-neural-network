# MNISC Digit Identification Neural Network
With this project I aim to better understand neural networks at their core.

## Code Explanation
As the code is mostly math, you can find what each one of the function does in relation to the neural network.

### Data loading
```
current_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(current_dir, 'train.csv'))
```
Here the training data is loaded form the the CSV file

```
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
```
This converts the data into a Numpy array, determines its dimensions(m, n) and shuffles the data randomly.
```
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1: n]
```
The data is separated into a smaller development set(used for validations) and a larger training set

### Neural Network Functions
#### Init Params
This initializes the weights(W1, W2) and biases(b1, b2) of the neural network
```
def init_params():
  W1 = np.random.rand(10, 784)
  b1 = np.random.rand(10, 1)
  W2 = np.random.rand(10, 10)
  b2 = np.random.rand(10, 1)
  return W1, b1, W2, b2
```
#### Rectified Linear Unit
This implements the Rectified Linear Unit(or ReLU) activation function. ReLU introduces non-linearity into the network, which allows it to learn complex patterns.
```
def ReLU(Z):
  return p.maximum(0, Z)
```
#### Derivative of the Rectified Linear Unit
Caculates the derivative of the ReLU function, which is really important for backpropagation(or calculation gradients). The derivative is 1 for positive inputs and 0 for negative ones 
```
def ReLU_deriv(Z):
    return Z > 0
```
#### Softmax
Implements the softmax activation function, used in the output layer for multi-class classification. It converts a a vector of raw output scores into a probability distribution over the classes
```
def softmax(Z):
  return np.exp(Z) / np.sum(np.exp(Z))

```
#### One Hot
This function converts categorical labels(0, 1, 2...) into one-hot encoded vectors([1, 0, 0], [0, 1, 0]...). This is fundamental as the neural network outputs probability over each class
```
def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arrange(Y.size, Y )] = 1
  one_hot_Y = one_hot_Y.T
  return one_hot_Y
```
#### Forward Propagation
Performs the foward propagation step, calculating the network's output for a given input
```
def forward_prop(W1, b1, W2, b2, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2
```
Calculates the weighted sum of inputs and adds biases at each layer (Z1, Z2).

Applies the ReLU activation function in the hidden layer (A1).

Applies the softmax activation function in the output layer (A2).

#### Backwards Propagation
Performs the backwards propagation algorithm, calculating the gradients of the loss function taking the weights and biases in consideration
```
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  db2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
  dW1 = 1 / m * dZ1.dot(X.T)
  db1 = 1 / m * np.sum(dZ1)
  return dW1, db1, dW2, db2
```
Computes the error signal at the output layer (dZ2).

Propagates the error back through the network, calculating the gradients for weights and biases (dW2, db2, dW1, db1).

#### Update Params
Updates the weights and biases using gradient descent.
```
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
  W1 = W1 - alpha * dW1
  b1 = b1 - alpha * db1    
  W2 = W2 - alpha * dW2  
  b2 = b2 - alpha * db2    
  return W1, b1, W2, b2
```
alpha: The learning rate, controlling the size of the update steps.

The weights and biases are adjusted in the direction that minimizes the loss function.

More functions to come... 