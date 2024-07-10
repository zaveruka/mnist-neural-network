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
  Z_max = np.max(Z, axis=0, keepdims=True)
  shifted_Z = Z - Z_max
  return np.exp(shifted_Z) / np.sum(np.exp(shifted_Z), axis=0, keepdims=True)


```
1. np.exp(Z): Calculates the element-wise exponential of the input Z.

2. np.sum(np.exp(Z)): Calculates the sum of the exponentials, which acts as a normalization factor.

3. The division normalizes the exponentials, producing a probability distribution over the output classes.

#### One Hot
This function converts categorical labels(0, 1, 2...) into one-hot encoded vectors([1, 0, 0], [0, 1, 0]...). This is essential for neural networks, as they output probabilities for each class. For example, label '2' becomes [0, 0, 1, 0] and so on.
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
1. Z1 = W1.dot(X) + b1: Calculates the weighted sum of inputs (X) and the weights of the first layer (W1) and adds the bias (b1).

2. A1 = ReLU(Z1): Applies the ReLU activation function to the result (Z1) to introduce non-linearity.

3. Z2 = W2.dot(A1) + b2: Calculates the weighted sum at the second layer (output layer), using the activated output from the first layer (A1), weights (W2), and bias (b2).

4. A2 = softmax(Z2): Applies the softmax function to the output of the second layer (Z2) to obtain a probability distribution over the output classes.

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
1. one_hot_Y = one_hot(Y): Converts the true labels (Y) into one-hot encoded vectors.

2. dZ2 = A2 - one_hot_Y: Calculates the error signal at the output layer (dZ2) as the difference between the predicted probabilities (A2) and the one-hot encoded true labels.

3. dW2 = 1 / m * dZ2.dot(A1.T): Calculates the gradient of the loss function with respect to the weights of the second layer (dW2) using the error signal (dZ2) and the activated output of the first layer (A1).

4. db2 = 1 / m * np.sum(dZ2): Calculates the gradient of the loss function with respect to the bias of the second layer (db2).

5. dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1): Calculates the error signal at the hidden layer (dZ1) by backpropagating the error from the output layer (dZ2) through the weights of the second layer (W2) and multiplying by the derivative of the ReLU activation function.

6. dW1 = 1 / m * dZ1.dot(X.T) and db1 = 1 / m * np.sum(dZ1): Calculate the gradients of the loss function with respect to the weights (dW1) and bias (db1) of the first layer using the error signal at the hidden layer (dZ1) and the input data (X).

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
alpha: This is the learning rate, a hyperparameter that controls the step size of the updates.

The equations update the weights and biases by subtracting a fraction of their corresponding gradients, moving them in the direction that minimizes the loss function.

#### Get Predictions

```
def get_predictions(A2):
  return np.argmax(A2, 0)
```

Takes the output of the neural network's final layer (A2, which represents probabilities for each class) and returns the class with the highest probability using np.argmax(A2, 0).


#### Get Accuracy 

```
def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y) / Y.size
```
Calculates the accuracy of the model's predictions (predictions) against the true labels (Y).

#### Gradient Descent

```
def gradient_descent(X, Y, alpha, iterations):
  W1, b1, W2, b2 = init_params()
  for i in range(iterations):
      Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
      dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
      W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
      if i % 10 == 0:
          print("Iteration: ", i)
          predictions = get_predictions(A2)
          print(get_accuracy(predictions, Y))
  return W1, b1, W2, b2

```

This is the core training function.

1. Initializes weights (W1, W2) and biases (b1, b2) using init_params()

2. Iterates through the training data (iterations times):

3. Goes through each one of the previously explained functions, displaying each 10 iterations the current accuracy of the model

#### Data normalization

```
X_train = X_train / 255.0
X_dev = X_dev / 255.0 
```
This operation scales the pixel values of your images to a range between 0 and 1. This improves learning rates and model stability