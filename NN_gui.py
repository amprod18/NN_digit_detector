import time
import gzip
import os
import numpy as np
import custom_NN
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk


class Main_menu(ctk.CTk):
    def __init__(self):
        super().__init__()

        # self.overrideredirect(True) # Used for fullscreen later on
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.title("Neural Network Digit Identifier")
        self.geometry(f"{screen_width}x{screen_height}")
        self.geometry("+0+0")

        image_path = cwd + "images/neural_network_icon_dth.png"

        
        
        image = ctk.CTkImage(light_image=Image.open("images/neural_network_icon_lth.png"), dark_image=Image.open("images/neural_network_icon_dth.png"), size=(250, 250))
        image_width, image_height = 250, 250
        self.start_logo = ctk.CTkLabel(self, image=image, text="")
        x = (screen_width - image_width) // 2
        y = (screen_height - image_height) // 2

        self.start_label = ctk.CTkLabel(self, text="", width=screen_width, height=screen_height, fg_color='transparent')
        self.start_label.bind(command=lambda x, y: self.initial_progress_bar(x, y))
        self.start_label.place(x=0, y=0)

        self.start_logo.place(x=x, y=y)

        self.progressbar = ctk.CTkProgressBar(self, width=image_width, mode='determinate', determinate_speed=0.2)
        self.progressbar.set(0)
        y = y + image_height + 20
        
        self.start_text = ctk.CTkLabel(self, width=image_width, height=50, text="Press Anywhere to Continue", fg_color='transparent')
        self.start_text.place(x=x, y=y)

    def initial_progress_bar(self, x, y):
        self.start_text.destroy()
        self.start_label.destroy()
        self.progressbar.place(x=x, y=y)
        self.progressbar.start()

        prog = 0
        self.progressbar.start()
        
        while prog < 1:
            self.progressbar.set(prog)
            step = 2 * np.random.random_sample() / 5
            prog += step
            time.sleep(1.3 * np.random.random_sample() + 0.2)
            self.update_idletasks()
        self.progressbar.stop()
        self.progressbar.set(1)


             
             


def read_data(filename_images, filename_labels):
    cp1 = time.perf_counter()
    with gzip.open(cwd + filename_images, 'rb') as f:
            images = f.read()
    cp2 = time.perf_counter()
    images_time = str((cp2-cp1)*1000)

    cp1 = time.perf_counter()
    with gzip.open(cwd + filename_labels, 'rb') as f:
            labels = f.read()
    cp2 = time.perf_counter()
    labels_time = str((cp2-cp1)*1000)

    return (images, labels, images_time, labels_time)

def load_data(filename_images_train, filename_labels_train, filename_images_test, filename_labels_test):
    train_images, train_labels, train_images_time, train_labels_time = read_data(filename_images_train, filename_labels_train)
    test_images, test_labels, test_images_time, test_labels_time = read_data(filename_images_test, filename_labels_test)

    # Documentation on how to read the data here: http://yann.lecun.com/exdb/mnist/
    train_images = np.frombuffer(train_images, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    train_labels = np.frombuffer(train_labels, dtype=np.uint8, offset=8).reshape(-1)
    train_images = train_images.reshape(60000, 784).T / 255
    test_images = np.frombuffer(test_images, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    test_labels = np.frombuffer(test_labels, dtype=np.uint8, offset=8).reshape(-1)
    test_images = test_images.reshape(10000, 784).T / 255

    print('[INFO] Time elapsed reading train images:\t{0} ms\n[INFO] Time elapsed reading train labels:\t{1} ms\n'.format(train_images_time, train_labels_time))

    return (train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    global cwd 
    cwd = os.getcwd() + "\\"

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    Gui = Main_menu()

    Gui.mainloop()

    train_images, train_labels, test_images, test_labels = load_data("training_set/train-images-idx3-ubyte.gz", "training_set/train-labels-idx1-ubyte.gz", "test_set/t10k-images-idx3-ubyte.gz", "test_set/t10k-labels-idx1-ubyte.gz")

    learning_rate = 0.05
    N_network = custom_NN.neural_network([10, 10], learning_rate)
    # N_network.train_mode(train_images, train_labels)
    # N_network.save_params("params_file")
    N_network.retrieve_params(cwd + "params_file")
    data = np.random.randint(0, 1e4)
    N_network.predict_mode(test_images[:, data], test_labels[data])
