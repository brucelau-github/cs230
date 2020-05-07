""" machine learning model """
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset

def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True,
          initialization="he", optimizer="adam"):
    """ build """
    grads = {}
    costs = []
    layers_dims = [X.shape[0], 512, 128, 64, 4]

    initilization_dict = {
        "zero": initialize_parameters_zero,
        "random": initialize_parameters_random,
        "he": initialize_parameters_he,
        "xavier": initialize_parameters_xavier
    }
    optimizers = {
        "adam": adam_optimizer
    }

    parameters = initilization_dict[initialization](layers_dims)

    for i in num_iterations:
        A3, caches = forward_propagation(X, parameters)
        cost = cross_entropy(A3, Y)

        grads = backward_propagation(X, Y, caches)

        parameters = optimizers[optimizer](parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    return (parameters, costs)


def initialize_parameters_zero(layers_dims):
    """ zero initialization """
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def initialize_parameters_random(layers_dims):
    """ randomly initailize"""
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn((layers_dims[l], layers_dims[l-1]))
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def initialize_parameters_he(layers_dims):
    """ he initialization """
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def initialize_parameters_xavier(layers_dims):
    """ he initialization """
    L = len(layers_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn((layers_dims[l], layers_dims[l-1])) * np.sqrt(2/(layers_dims[l-1] + layers_dims[l]))
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def forward_propagation(X, parameters):
    """ forward propagation
    3 LINEAR->RELU -> LINEAR->SOFTMAX
    """
    caches = {"A0": X}
    L = 4

    for l in range(1, L):
        caches["Z"+str(l)] = np.dot(parameters["W"+str(l)], caches["A"+str(l-1)]) + parameters["b"+str(l)]
        caches["A"+str(l)] = relu(caches["Z"+str(l)])

    caches["Z4"] = np.dot(parameters["W4"], caches["A3"]) + parameters["b4"]
    caches["A4"] = softmax(caches["Z4"])

    return caches["A4"], caches

def relu(Z):
    """ relu """
    return np.maximum(0, Z)

def softmax(Z):
    """ soft max """
    expo = np.exp(Z)
    expo_sum = np.sum(np.exp(Z))
    return expo/expo_sum

def cross_entropy(yhat, y):
    """ compute lost function """
    cost = - np.mean(y * np.log(yhat + 1e-8))
    cost = float(np.squeeze(cost))
    assert(isinstance(cost, float))
    return cost

def backward_propagation(X, Y, caches):
    """ backward propagation """
    grads = {}

    m = X.shape[1]

    grads["dZ4"] = caches["A4"] - Y

    grads["dW4"] = np.dot(grads["dZ4"], caches["A3"].T) / m
    grads["db4"] = np.sum(grads["dZ4"], axis=1, keepdims=True) / m

    for l in reversed(range(1, 4)):
        grads["dA"+str(l)] = np.dot(caches["W"+str(l+1)].T, grads["Z"+str(l+1)])
        grads["dZ"+str(l)] = grads["dA"+str(l)] * np.int64(caches["A"+str(l)] > 0)
        grads["dW"+str(l)] = np.dot(grads["dZ"+str(l)], caches["A"+str(l-1)].T) / m
        grads["db"+str(l)] = np.sum(grads["dZ"+str(l)], axis=1, keepdims=True) / m

    return grads

def adam_optimizer(parameters, grads, learning_rate):
    """ adam optimizer """
    for l in range(1, 4):
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate * grads["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate * grads["db"+str(l)]

    return parameters

def predict(parameters, X, Y):
    """
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    yhat, _ = forward_propagation(X, parameters)
    predictions = yhat > 0.5

    return predictions

def train_model():
    """ train model """
    learning_rate = 0.005
    train_X, train_Y, test_X, test_Y = load_dataset()
    parameters, costs = model(train_X[:, :10000], train_Y[:, :100000], learning_rate)
    #print("On the train set:")
    #predictions_train = predict(train_X, train_Y, parameters)
    #print("On the test set:")
    #predictions_test = predict(test_X, test_Y, parameters)

    #print("predictions_train = " + str(predictions_train))
    #print("predictions_test = " + str(predictions_test))

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

train_model()
