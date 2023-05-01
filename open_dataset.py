import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import *


class neuron():
    def __init__(self, layer, pos):
        self.weight = np.random.random(0, 1)
        self.bias = np.random.random(0, 1)
        self.layer = layer
        self.pos = pos

    def update(self, weight=None, bias=None):
        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

def read_data(path):
    cp1 = time()
    with gzip.open(path, 'rb') as f:
            data = f.read()
    cp2 = time()
    return (data, str((cp2-cp1)*1000))

def nums2vects(labels):
    cols = np.zeros((labels.size, labels.max() + 1))
    cols[np.arange(labels.size), labels] = 1
    return cols.T

def show_image(image, label):
    fig, ax = plt.subplots(num=0)
    ax.axis("off")
    ax.set_title(label)
    ax.imshow(image, cmap='gray')
    plt.show()

def show_rand_image(images, labels):
    data_point = np.random.randint(0, 6e4)
    show_image(images[data_point], labels[data_point])

def create_NN(layers, neurons):
    weights = np.zeros(layers)
    biases = np.zeros(layers)

    if len(neurons) == layers + 1:
        for i in range(layers):
            weights[i] = np.random.randn(neurons[i + 1], neurons[i])
            biases[i] = np.random.randn(neurons[i + 1], 1)
    else:
        print("[ERROR] Number of layers and neurons numbers do not match.")

    return (weights, biases)

def relu(values):
    return np.maximum(0, values)

def relu_prime(values):
    return values > 0

def softmax(array):
    return (np.exp(array) / np.exp(array).sum())

def forward_prop(weights, biases, images):
    activations = np.zeros(len(weights))
    zs = np.zeros(len(weights))
    z = np.dot(weights[0], images) + biases[0]
    activation = relu(z)

    activations[0] = activation
    zs[0] = z

    for i, v in enumerate(weights):
        if i == 0:
            continue
        elif i == len(weights):
            continue
        else:
            z = np.dot(v, activation) + biases[i]
            activation = relu(z)

            activations[i] = activation
            zs[i] = z
    
    z = np.dot(weights[-1], activation) + biases[-1]
    activation = softmax(z)

    activations[-1] = activation
    zs[-1] = z

    return (weights, zs, activations)

def backward_prop(weights, zs, activations, images, labels):
    objectives = nums2vects(labels)
    dA0 = 2*(activations[-1, :] - objectives)
    dA1 = dA0*relu_prime(zs[-1, :])*weights[-1]

    dws = np.zeros(len(weights))
    dbs = np.zeros(len(weights))
    dAs = np.zeros(len(activations))
    dAs[0] = dA0
    dAs[1] = dA1

    for i in range(len(weights)):
        if (i == 0) or (i == 1):
            continue
        elif i == len(weights):
            continue
        else:
            dw1 = dA1*relu_prime(zs[-i, :])*activations[-i+1, :]
            db1 = dA1*relu_prime(zs[-i, :])
            dA1 = dA1*relu_prime(zs[-i, :])*weights[-i]
            dAs[-i] = dA1
            dws[-i] = dw1
            dbs[-i] = db1

    dw1 = dA1*relu_prime(images)*activations[1, :]
    db1 = dA1*relu_prime(images)
    dA1 = dA1*relu_prime(images)*weights[0]
    dAs[0] = dA1
    dws[0] = dw1
    dbs[0] = db1

    return (dws, dbs)

def update_NN(weights, biases, dws, dbs, learning_rate):
    weights = weights - learning_rate * dws
    biases = biases - learning_rate * dbs
    return (weights, biases)

def gradient_descend(images, labels, learning_rate, iters):
    layers = 2
    neurons = np.array([784, 10, 10])
    weights, biases = create_NN(layers, neurons)

    for i in range(iters):
        weights, zs, activations = forward_prop(weights, biases, images)
        dws, dbs = backward_prop(weights, zs, activations, images, labels)
        weights, biases = update_NN(weights, biases, dws, dbs, learning_rate)
        if i % 50 == 0:
            print("[INFO] iteration number: ", i)
            print("[INFO] Accuracy: ", get_accuracy(get_predictions(activations[-1]), labels))
    return (weights, biases)

def get_predictions(pred):
    return np.argmax(pred, 0)

def get_accuracy(pred, labels):
    return np.sum(pred == labels) / labels.size


if __name__ == '__main__':
    path2images = "training_set/train-images-idx3-ubyte.gz"
    images, img_time = read_data(path2images)

    path2labels = "training_set/train-labels-idx1-ubyte.gz"
    labels, lbl_time = read_data(path2labels)

    # Documentation on how to read the data here: http://yann.lecun.com/exdb/mnist/
    images = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.frombuffer(labels, dtype=np.uint8, offset=8).reshape(-1)

    print('Time elapsed reading images:\t{0} ms\nTime elapsed reading labels:\t{1} ms\n'.format(img_time, lbl_time))

    # show_rand_image(images, labels)

    weights, biases = gradient_descend(images, labels, 0.05, 100)
