import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

def plot_accuracy(accuracy, iters):
    fig, ax = plt.subplots(num=0)
    ax.set_title("Number of Iterations: " + str(iters))
    ax.set_xlabel("Number of Iteration")
    ax.set_ylabel("Accuracy Achieved")
    ax.plot(range(1, iters + 1), accuracy)
    plt.show()

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

def sigmoid(values):
    return 1/(1+np.exp(-values))

def sigmoid_prime(values):
    return (sigmoid(values)*(1-sigmoid(values)))

def softmax(array):
    ar = np.exp(array) / np.sum(np.exp(array))
    # print(ar, np.sum(ar))
    return (np.exp(array) / np.sum(np.exp(array)))

def forward_prop(weights, biases, images):
    cp1 = time()
    activations = []
    zs = []
    z = np.dot(weights[0], images.reshape(-1, 1)) + biases[0].reshape(-1, 1)
    activation = sigmoid(z)
    activations.append(activation)
    zs.append(z)

    for i, v in enumerate(weights):
        if i == 0:
            continue
        elif i == len(weights) - 1:
            continue
        else:
            z = np.dot(v, activation) + biases[i].reshape(-1, 1)
            activation = sigmoid(z)

            activations.append(activation)
            zs.append(z)
    
    z = np.dot(weights[-1], activation) + biases[-1].reshape(-1, 1)
    activation = softmax(z)

    activations.append(activation)
    zs.append(z)
    cp2 = time()

    return (weights, zs, activations, 1000 * (cp2 - cp1))

def backward_prop(weights, zs, activations, images, objective):
    cp1 = time()
    dws = []
    dbs = []
    dAs = []

    dA0 = 2*(activations[-1] - objective.reshape(-1, 1))
    dAs.append(dA0)

    db1 = dA0 * sigmoid_prime(zs[-1]).reshape(-1, 1)
    dA1 = np.dot(weights[-1].T, db1.reshape(-1, 1))
    dw1 = np.dot(db1.reshape(-1, 1), activations[-2].reshape(-1, 1).T)
    
    dAs.append(dA1)
    dbs.append(db1)
    dws.append(dw1)
    
    for i in range(len(weights) + 1):
        if (i == 0) or (i == 1):
            continue
        elif i == len(weights):
            continue
        else:
            db1 = dA1 * sigmoid_prime(zs[-i]).reshape(-1, 1)
            dw1 = np.dot(db1.reshape(-1, 1), activations[-i-1].reshape(-1, 1).T)
            dA1 = np.dot(weights[-i].T, db1.reshape(-1, 1))
            dAs.insert(0, dA1)
            dws.insert(0, dw1)
            dbs.insert(0, db1)

    db1 = dA1 * sigmoid_prime(zs[0]).reshape(-1, 1)
    dw1 = np.dot(db1.reshape(-1, 1), images.reshape(-1, 1).T)
    dA1 = np.dot(weights[0].T, db1.reshape(-1, 1))
    dAs.insert(0, dA1)
    dws.insert(0, dw1)
    dbs.insert(0, db1)
    cp2 = time()

    return (dws, dbs, 1000 * (cp2 - cp1))

def update_NN(weights, biases, dws, dbs, learning_rate):
    cp1 = time()
    u_weights = []
    u_biases = []
    for i,v in enumerate(weights):
        u_weights.append(v - learning_rate * dws[i])
        u_biases.append(biases[i].reshape(-1, 1) - learning_rate * dbs[i])
    cp2 = time()
    
    return (u_weights, u_biases, 1000 * (cp2 - cp1))

def gradient_descend(images, labels, learning_rate, iters):
    cp1 = time()
    layers = 2
    neurons = np.array([784, 10, 10, 10])
    weights, biases = create_NN(layers, neurons)
    objectives = nums2vects(labels)
    counter = 0
    accuracy = np.array([])
    times = np.zeros((3, iters))

    for i in range(iters):
        weights, zs, activations, fp_time = forward_prop(weights, biases, images[:, i])
        dws, dbs, bp_time = backward_prop(weights, zs, activations, images[:, i], objectives[:, i])
        weights, biases, unn_time = update_NN(weights, biases, dws, dbs, learning_rate)
        counter += (np.argmax(activations[-1], 0) == labels[i])[0]
        times[:, i] = np.array([fp_time, bp_time, unn_time])
        if i == 0:
            continue
        accuracy = np.append(accuracy, 100*counter / i)
        if i % 10000 == 0:
            print("[INFO] iteration number: ", i)
            print("[INFO] Accuracy: ", round(100*counter / i, 2), "%")
    print("[INFO] iteration number: ", iters)
    print("[INFO] Accuracy: ", round(100*counter / iters, 2), "%")
    accuracy = np.append(accuracy, 100*counter / iters)
    times = np.array([i.sum()/iters for i in times])
    times = np.append(times, times.sum())
    
    cp2 = time()
    mean_times = np.append(times, np.array([1000 * (cp2 - cp1)]))
    return (weights, biases, accuracy, mean_times)

if __name__ == '__main__':
    start_cp = time()
    path2images = "training_set/train-images-idx3-ubyte.gz"
    images, img_time = read_data(path2images)

    path2labels = "training_set/train-labels-idx1-ubyte.gz"
    labels, lbl_time = read_data(path2labels)

    # Documentation on how to read the data here: http://yann.lecun.com/exdb/mnist/
    images = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    labels = np.frombuffer(labels, dtype=np.uint8, offset=8).reshape(-1)
    data_images = images.reshape(60000, 784).T / 255

    print('[INFO] Time elapsed reading images:\t{0} ms\n[INFO] Time elapsed reading labels:\t{1} ms\n'.format(img_time, lbl_time))

    # show_rand_image(images, labels)
    learning_rate, iters = 0.05, 60000

    weights, biases, accuracy, mean_times = gradient_descend(data_images, labels, learning_rate, iters)
    end_cp = time()
    mean_times = np.append(mean_times, 1000*(end_cp - start_cp))
    table_headers = np.array([[""], ["Mean Time per Iteration (ms)"], ["Iterations per Second"]])
    time_names = np.array(["Forward Propagation", "Backward Propagation", "Update Neural Network", "Total Iteration", "Gradient Descend", "Total Program"])
    time_data = np.column_stack((table_headers, np.array([time_names, mean_times, 1000/mean_times]))).T

    # Create figure and grid layout
    fig = plt.figure(num=1)
    gs = GridSpec(1, 2, width_ratios=[2, 1])  # Divide the figure into 1 row, 2 columns

    # Create the left subplot for the table
    ax_table = fig.add_subplot(gs[0, 0])
    table = ax_table.table(cellText=time_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    ax_table.axis('off')

    # Create the right subplot for the graphic
    ax_graphic = fig.add_subplot(gs[0, 1])
    ax_graphic.set_title("Number of Iterations: " + str(iters))
    ax_graphic.set_xlabel("Number of Iteration")
    ax_graphic.set_ylabel("Accuracy Achieved")
    ax_graphic.plot(range(1, iters + 1), accuracy)

    # Adjust spacing between subplots
    fig.tight_layout()
    plt.show()
