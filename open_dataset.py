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
    weights = []
    biases = []
    
    if len(neurons) == layers + 2:
        for i in range(layers + 1):
            weights.append(np.random.randn(neurons[i + 1], neurons[i]))
            biases.append(np.random.randn(neurons[i + 1]).T)
    else:
        print("[ERROR] Number of layers and neurons numbers do not match.")

    return (weights, biases)

def relu(values):
    return np.maximum(0, values)

def relu_prime(values):
    return values > 0

def softmax(array):
    return (np.exp(array) / np.sum(np.exp(array)))

def forward_prop(weights, biases, images):
    activations = []
    zs = []
    z = np.dot(weights[0], images) + biases[0]
    activation = relu(z)

    activations.append(activation)
    zs.append(z)

    for i, v in enumerate(weights):
        if i == 0:
            continue
        elif i == len(weights) - 1:
            continue
        else:
            z = np.dot(v, activation) + biases[i]
            activation = relu(z)

            activations.append(activation)
            zs.append(z)
    
    z = np.dot(weights[-1], activation) + biases[-1]
    activation = softmax(z)

    activations.append(activation)
    zs.append(z)
    # print("-----BIASES-----\n", biases)

    return (weights, zs, activations)

def backward_prop(weights, zs, activations, images, objective):
    # print("-----WEIGHTS-----\n", weights)
    # print("-----ACTIVATIONS-----\n", activations)
    
    dA0 = 2*(activations[-1] - objective)
    dA1 = np.dot(np.dot(dA0, relu_prime(zs[-1])), weights[-1])

    dws = []
    dbs = []
    dAs = []
    dAs.append(dA0)
    dAs.append(dA1)

    for i in range(len(weights) + 1):
        if (i == 0) or (i == 1):
            continue
        else:
            db1 = np.dot(dA1, relu_prime(zs[-i]))
            dw1 = np.dot(db1, activations[-i+1])
            dA1 = np.dot(db1, weights[-i])
            dAs.insert(0, dA1)
            dws.insert(0, dw1)
            dbs.insert(0, db1)

    db1 = np.dot(dA1, relu_prime(images))
    dw1 = np.dot(db1, activations[1])
    dA1 = np.dot(db1, weights[0])
    dAs.insert(0, dA1)
    dws.insert(0, dw1)
    dbs.insert(0, db1)

    return (dws, dbs)

def update_NN(weights, biases, dws, dbs, learning_rate):
    u_weights = []
    u_biases = []
    for i,v in enumerate(weights):
        u_weights.append(v - learning_rate * dws[i])
        u_biases.append(biases[i] - learning_rate * dbs[i])
    return (u_weights, u_biases)

def gradient_descend(images, labels, learning_rate, iters):
    layers = 2
    neurons = np.array([784, 10, 10, 10])
    weights, biases = create_NN(layers, neurons)
    objectives = nums2vects(labels)

    for i in range(iters):
        weights, zs, activations = forward_prop(weights, biases, images[:, i])
        dws, dbs = backward_prop(weights, zs, activations, images[:, i], objectives[:, i])
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
    data_images = images.reshape(60000, 784).T / 255

    print('Time elapsed reading images:\t{0} ms\nTime elapsed reading labels:\t{1} ms\n'.format(img_time, lbl_time))

    # show_rand_image(images, labels)

    weights, biases = gradient_descend(data_images, labels, 0.05, 100)
