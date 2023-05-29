import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import pickle
import os


class neural_network():
    def __init__(self, layers_size, learning_rate):
        layers_size.insert(0, 784)
        layers_size.insert(-1, 10)
        self.layers_size = layers_size
        self.learning_rate = learning_rate
        self.accuracy = 0
        self.real_accuracy = 0
        self.weights = []
        self.biases = []
        self.zs = []
        self.activations = []

        for i in range(len(self.layers_size) - 1):
            self.weights.append(np.random.randn(self.layers_size[i + 1], self.layers_size[i]))
            self.biases.append(np.random.randn(self.layers_size[i + 1]).T)
            self.zs.append(np.zeros(self.layers_size[i]).T)
            self.activations.append(np.zeros(self.layers_size[i]).T)
        
        self.dws = self.weights.copy()
        self.dbs = self.biases.copy()
        self.dAs = self.activations.copy()
        
    def forward_prop(self, image):
        cp1 = time.perf_counter()
        z = np.dot(self.weights[0], image.reshape(-1, 1)) + self.biases[0].reshape(-1, 1)
        activation = self.sigmoid(z)
        self.activations[0] = activation
        self.zs[0] = z

        for i, v in enumerate(self.weights):
            if i == 0:
                continue
            elif i == len(self.weights) - 1:
                continue
            else:
                z = np.dot(v, activation) + self.biases[i].reshape(-1, 1)
                activation = self.sigmoid(z)

                self.activations[i] = activation
                self.zs[i] = z
        
        z = np.dot(self.weights[-1], activation) + self.biases[-1].reshape(-1, 1)
        activation = self.softmax(z)

        self.activations[-1] = activation
        self.zs[-1] = z
        cp2 = time.perf_counter()

        return (1000 * (cp2 - cp1))
    
    def backward_prop(self, image, objective):
        cp1 = time.perf_counter()

        dA0 = 2*(self.activations[-1] - objective.reshape(-1, 1))
        self.dAs[-1] = dA0

        db1 = dA0 * self.sigmoid_prime(self.zs[-1]).reshape(-1, 1)
        dA1 = np.dot(self.weights[-1].T, db1.reshape(-1, 1))
        dw1 = np.dot(db1.reshape(-1, 1), self.activations[-2].reshape(-1, 1).T)
        
        self.dAs[-2] = dA1
        self.dbs[-1] = db1
        self.dws[-1] = dw1
        
        for i in range(len(self.weights) + 1):
            if (i == 0) or (i == 1):
                continue
            elif i == len(self.weights):
                continue
            else:
                db1 = dA1 * self.sigmoid_prime(self.zs[-i]).reshape(-1, 1)
                dw1 = np.dot(db1.reshape(-1, 1), self.activations[-i-1].reshape(-1, 1).T)
                dA1 = np.dot(self.weights[-i].T, db1.reshape(-1, 1))
                self.dAs[-i-1] = dA1
                self.dws[-i] = dw1
                self.dbs[-i] = db1

        db1 = dA1 * self.sigmoid_prime(self.zs[0]).reshape(-1, 1)
        dw1 = np.dot(db1.reshape(-1, 1), image.reshape(-1, 1).T)
        dA1 = np.dot(self.weights[0].T, db1.reshape(-1, 1))
        self.dAs[0] = dA1
        self.dws[0] = dw1
        self.dbs[0] = db1
        cp2 = time.perf_counter()

        return (1000 * (cp2 - cp1))
    
    def update_NN(self):
        cp1 = time.perf_counter()
        for i,v in enumerate(self.weights):
            self.weights[i] = v - self.learning_rate * self.dws[i]
            self.biases[i] = self.biases[i].reshape(-1, 1) - self.learning_rate * self.dbs[i]
        cp2 = time.perf_counter()
        
        return (1000 * (cp2 - cp1))
    
    def gradient_descend(self, images, labels):
        cp1 = time.perf_counter()
        objectives = self.nums2vects(labels)
        counter = 0
        accuracy = np.array([])
        iters = len(images[0])
        times = np.zeros((3, iters))

        for i in range(len(images[0])):
            fp_time = self.forward_prop(images[:, i])
            bp_time = self.backward_prop(images[:, i], objectives[:, i])
            unn_time = self.update_NN()
            counter += (np.uint8(np.argmax(self.activations[-1], 0)[0]) == labels[i])
            times[:, i] = np.array([fp_time, bp_time, unn_time])
            if i == 0:
                continue
            accuracy = np.append(accuracy, 100*counter / i)
        accuracy = np.append(accuracy, 100*counter / iters)
        times = np.array([i.sum()/iters for i in times])
        times = np.append(times, times.sum())
        
        cp2 = time.perf_counter()
        mean_times = np.append(times, np.array([1000 * (cp2 - cp1)]))
        return (accuracy, mean_times)
    
    def sigmoid(self, values):
        return 1/(1+np.exp(-values))

    def sigmoid_prime(self, values):
        return (self.sigmoid(values)*(1-self.sigmoid(values)))

    def softmax(self, array):
        return (np.exp(array) / np.sum(np.exp(array)))
    
    def nums2vects(self, labels):
        cols = np.zeros((labels.size, labels.max() + 1))
        cols[np.arange(labels.size), labels] = 1
        return cols.T
    
    def show_image(self, image, title, fignum):
        fig, ax = plt.subplots(num=fignum)
        ax.axis("off")
        ax.set_title(title)
        ax.imshow(image, cmap='gray')
        plt.show()
    
    def plot_training_data(self, accuracy, time_data):
        # Create figure and grid layout
        fig = plt.figure(num=0)
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
        ax_graphic.set_title("Number of Iterations: " + str(len(accuracy)))
        ax_graphic.set_xlabel("Number of Iteration")
        ax_graphic.set_ylabel("Accuracy Achieved")
        ax_graphic.plot(range(1, len(accuracy) + 1), accuracy)

        # Adjust spacing between subplots
        fig.tight_layout()
        plt.show()
    
    def train_mode(self, images, labels, show=False):
        accuracy, mean_times = self.gradient_descend(images, labels)
        self.accuracy = accuracy[-1]
        table_headers = np.array([[""], ["Mean Time per Iteration (ms)"], ["Iterations per Second"]])
        self.time_names = np.array(["Forward Propagation", "Backward Propagation", "Update Neural Network", "Total Iteration", "Total Training"])
        self.time_data = np.column_stack((table_headers, np.array([self.time_names, mean_times, 1000/mean_times]))).T
        if show:
            self.plot_training_data(accuracy, self.time_data)
    
    def test_mode(self, images, labels):
        self.real_accuracy = 0
        self.mean_pred_time = 0
        for i in range(len(images[0])):
            pred, fp_time = self.predict_mode(images[:, i], labels[i])
            self.mean_pred_time += fp_time
            self.real_accuracy += (np.uint8(pred[0]) == labels[i])

        self.mean_pred_time /= len(images[0])
        self.real_accuracy /= len(images[0])
        self.mean_pred_time = round(1000*self.mean_pred_time, 2)
        self.real_accuracy = round(100*self.real_accuracy, 2)
        return (self.real_accuracy, self.mean_pred_time)

    def predict_mode(self, image, label, show=False):
        fp_time = self.forward_prop(image)
        pred = np.argmax(self.activations[-1], 0)
        if show:
            self.show_image(image.reshape(28, 28), f"Predicted number: {pred[0]}   Correct label: {label}", 1)
        return (pred, fp_time)
    
    def save_params(self, output_file):
        os.makedirs(os.path.dirname(f"{output_file}\\params\\biases.txt"), exist_ok=True)
        os.makedirs(os.path.dirname(f"{output_file}\\params\\weights.txt"), exist_ok=True)
        os.makedirs(os.path.dirname(f"{output_file}\\params\\metadata.txt"), exist_ok=True)
        with open(f"{output_file}\\params\\biases.txt", "wb") as f:
            pickle.dump(self.biases, f)
        with open(f"{output_file}\\params\\weights.txt", "wb") as f:
            pickle.dump(self.weights, f)
        with open(f"{output_file}\\params\\metadata.txt", "wb") as f:
            pickle.dump([self.real_accuracy, self.mean_pred_time, self.learning_rate], f)

    def retrieve_params(self, params_file):
        with open(f"{params_file}\\params\\biases.txt", "rb") as f:
            self.biases = pickle.load(f)
            self.activations = self.biases.copy()
            self.zs = self.biases.copy()
        with open(f"{params_file}\\params\\weights.txt", "rb") as f:
            self.weights = pickle.load(f)
        with open(f"{params_file}\\params\\metadata.txt", "rb") as f:
            self.real_accuracy, self.mean_pred_time, self.learning_rate = tuple(pickle.load(f))
        
        self.layers_size = [len(i[0]) for i in self.weights]



