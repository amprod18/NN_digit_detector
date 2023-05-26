import time
import gzip
import os
import numpy as np
import custom_NN
import customtkinter as ctk
from PIL import Image
import keyboard


class Main_menu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0
        self.delta = 0.05

        self.overrideredirect(True) # Used for fullscreen later on
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.font = ctk.CTkFont(family="helvetica", size=self.screen_height//45)
        self.title("Neural Network Digit Identifier")
        self.geometry(f"{self.screen_width}x{self.screen_height}")
        self.geometry("+0+0")
        self.main_frame = ctk.CTkTabview(self, width=self.screen_width-20, height=self.screen_height-40, state='disabled')
        self.main_frame.grid(row=0, column=0, padx=(10, 10), pady=(30, 10), sticky="nsew")
        self.main_frame.add("Loading Screen")
        self.main_frame.add("Main Menu")
        self.exit_button_image = ctk.CTkImage(light_image=Image.open("images/exit_button.png"), size=(20, 20))
        self.exit_button = ctk.CTkButton(self, text="", width=20, height=20, image=self.exit_button_image, corner_radius=5, fg_color="#F57A91", hover_color='#FF002C')
        self.exit_button.configure(command=self.exit_app)
        self.exit_button.place(x=self.screen_width-45, y=10)

        self.logo_image = ctk.CTkImage(light_image=Image.open("images/neural_network_icon_lth.png"), dark_image=Image.open("images/neural_network_icon_dth.png"), size=(20, 20))
        self.logo_label = ctk.CTkLabel(self, text="", width=20, height=20, image=self.logo_image, corner_radius=5)
        self.logo_label.place(x=10, y=10)
        
        self.logo = ctk.CTkImage(light_image=Image.open("images/neural_network_icon_lth.png"), dark_image=Image.open("images/neural_network_icon_dth.png"), size=(250, 250))
        image_width, image_height = 250, 250
        self.start_logo = ctk.CTkLabel(self.main_frame.tab("Loading Screen"), image=self.logo, text="")
        x = (self.screen_width - 20 - image_width) // 2
        y = (self.screen_height - 40 - image_height) // 2

        self.start_logo.place(x=x, y=y)

        self.progressbar = ctk.CTkProgressBar(self.main_frame.tab("Loading Screen"), width=image_width, mode='determinate', determinate_speed=0.2)
        self.progressbar.set(0)
        y = y + image_height + 30
        
        self.start_text = ctk.CTkLabel(self.main_frame.tab("Loading Screen"), width=image_width, height=50, text="Press SPACE to Continue", fg_color='transparent', font=self.font)
        self.start_text.place(x=x-27, y=y)

        if keyboard.is_pressed('space'):
            self.initial_progress_bar(x, y)

    def initial_progress_bar(self, x, y):
        self.progressbar.place(x=x, y=y+50)
        self.progressbar.start()

        prog = 0
        self.progressbar.start()
        
        while prog < 1:
            self.progressbar.set(prog)
            step = 0.1*np.random.random_sample()
            prog += step
            time.sleep(0.05*np.random.random_sample())
            self.update_idletasks()
        self.progressbar.stop()
        self.progressbar.set(1)
        
        self.start_logo.destroy()
        self.progressbar.destroy()

        self.main_menu(np.zeros(10), [None, 0])
    
    def main_menu(self, probabilities, pred):
        # Grid is 14x8
        rows, columns = tuple(range(14)), tuple(range(8))
        self.main_frame.tab("Main Menu").grid_columnconfigure(columns, weight=1)
        self.main_frame.tab("Main Menu").grid_rowconfigure(rows, weight=1)

        # Configure Input Zone (left column)
        self.load_model_button = ctk.CTkButton(self.main_frame.tab("Main Menu"), text="Load Model", font=self.font)
        self.train_model_button = ctk.CTkButton(self.main_frame.tab("Main Menu"), text="Train Model", font=self.font)
        self.load_model_button.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew", rowspan=2)
        self.train_model_button.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew", rowspan=2)
        self.train_model_button.configure(command=self.train_model)
        self.load_model_button.configure(command=self.load_model)

        self.NN_info_frame = ctk.CTkFrame(self.main_frame.tab("Main Menu"))
        self.NN_info_frame.grid(row=3, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew", columnspan=2, rowspan=10)
        self.NN_info_frame = NN_info_frame(self.NN_info_frame, self.font)

        self.predict_button = ctk.CTkButton(self.main_frame.tab("Main Menu"), text="Predict", font=self.font)
        self.predict_button.grid(row=13, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew", columnspan=2, rowspan=2)
        self.predict_button.configure(command=self.predict)

        # Configure NN Zone (center column)
        self.NN_image_frame = ctk.CTkFrame(self.main_frame.tab("Main Menu"))
        self.NN_image_frame.grid(row=1, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew", columnspan=4, rowspan=12)
        self.NN_image = ctk.CTkLabel(self.NN_image_frame, text="Neural Network image will appear here", fg_color='green', font=self.font)
        self.NN_image.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Configure Output Zone (right column)
        self.input_image = ctk.CTkLabel(self.main_frame.tab("Main Menu"), text="Input image will appear here", fg_color='green', font=self.font)
        self.input_image.grid(row=0, column=6, padx=(20, 20), pady=(20, 20), sticky="nsew", columnspan=2, rowspan=6)

        self.output_info_frame = ctk.CTkFrame(self.main_frame.tab("Main Menu"))
        self.output_info_frame.grid(row=7, column=6, padx=(20, 20), pady=(20, 20), sticky="nsew", columnspan=2, rowspan=7)
        self.output_info_frame = NN_output_frame(self.output_info_frame, probabilities, pred, self.font)

        self.main_frame.set("Main Menu")
    
    def train_model(self):
        learning_rate = float(self.NN_info_frame.learning_rate_entry.get())
        self.hidden_layers_sizes = [int(i) for i in self.NN_info_frame.hidden_layers_size_entry.get().split(sep=', ')]

        self.N_network = custom_NN.neural_network(self.hidden_layers_sizes, learning_rate)
        self.N_network.train_mode(train_images, train_labels)
        values = [learning_rate, self.hidden_layers_sizes, 784, 10, round(self.N_network.accuracy, 2), 'None']
        self.NN_info_frame.update_input_frame(values)
    
    def load_model(self):
        learning_rate = float(self.NN_info_frame.learning_rate_entry.get())
        self.hidden_layers_sizes = [int(i) for i in self.NN_info_frame.hidden_layers_size_entry.get().split(sep=', ')]

        self.N_network = custom_NN.neural_network(self.hidden_layers_sizes, learning_rate)
        self.N_network.retrieve_params(cwd + "params_file")
        values = [self.N_network.learning_rate, self.hidden_layers_sizes, 784, 10, self.N_network.accuracy, 'None']
        self.NN_info_frame.update_input_frame(values)

    def predict(self):
        pred_image = self.NN_info_frame.input_path_entry.get()

        data = np.random.randint(0, 1e4)
        pred, pred_time = self.N_network.predict_mode(test_images[:, data], test_labels[data])

        pred_image = ctk.CTkImage(light_image=Image.fromarray(test_images[:, data].reshape(28, 28)), size=(300, 300))
        self.input_image.configure(image=pred_image, text='')

        values = [self.N_network.learning_rate, self.hidden_layers_sizes, 784, 10, self.N_network.accuracy, pred_time]
        self.NN_info_frame.update_input_frame(values)

        self.output_info_frame.update_output_frame(self, self.N_network.activations[-1].reshape(1, 10), (pred, self.N_network.activations[-1][pred]))
    
    def exit_app(self):
        time.sleep(0.3)
        self.destroy()


class NN_info_frame(ctk.CTkFrame):
    def __init__(self, master, font):
        super().__init__(master)

        rows, columns = tuple(range(7)), tuple(range(2))
        self.master.grid_columnconfigure(columns, weight=1)
        self.master.grid_rowconfigure(rows, weight=1)

        self.frame_title = ctk.CTkLabel(master, text='Model Info', font=font)
        self.frame_title.grid(row=0, column=0, padx=(20, 20), pady=(20, 10), columnspan=2)
        # Hid layers 
        self.learning_rate_label = ctk.CTkLabel(master, text='Learning Rate: ', font=font)
        self.learning_rate_label.grid(row=1, column=0, padx=(20, 10), pady=(10, 10))
        self.learning_rate_entry = ctk.CTkEntry(master, placeholder_text='Sample: 0.05', font=font)
        self.learning_rate_entry.grid(row=1, column=1, padx=(10, 20), pady=(10, 10), sticky="ew")

        # Hid layers sizes
        self.hidden_layers_size_label = ctk.CTkLabel(master, text='Sizes of Hidden Layers: ', font=font)
        self.hidden_layers_size_label.grid(row=2, column=0, padx=(20, 10), pady=(10, 10))
        self.hidden_layers_size_entry = ctk.CTkEntry(master, placeholder_text='Sample: 10, 10', font=font)
        self.hidden_layers_size_entry.grid(row=2, column=1, padx=(10, 20), pady=(10, 10), sticky="ew")

        # Input size
        self.input_size_label = ctk.CTkLabel(master, text='Input Size: ', font=font)
        self.input_size_label.grid(row=3, column=0, padx=(20, 10), pady=(10, 10))
        self.input_size_entry = ctk.CTkLabel(master, text='None', font=font)
        self.input_size_entry.grid(row=3, column=1, padx=(10, 20), pady=(10, 10))

        # Output size
        self.output_size_label = ctk.CTkLabel(master, text='Output Size: ', font=font)
        self.output_size_label.grid(row=4, column=0, padx=(20, 10), pady=(10, 10))
        self.output_size_entry = ctk.CTkLabel(master, text='None', font=font)
        self.output_size_entry.grid(row=4, column=1, padx=(10, 20), pady=(10, 10))

        # Accuracy
        self.accuracy_label = ctk.CTkLabel(master, text='Accuracy: ', font=font)
        self.accuracy_label.grid(row=5, column=0, padx=(20, 10), pady=(10, 10))
        self.accuracy_entry = ctk.CTkLabel(master, text='None', font=font)
        self.accuracy_entry.grid(row=5, column=1, padx=(10, 20), pady=(10, 10))

        # Mean pred time
        self.pred_time_label = ctk.CTkLabel(master, text='Mean Prediction Time: ', font=font)
        self.pred_time_label.grid(row=6, column=0, padx=(20, 10), pady=(10, 20))
        self.pred_time_entry = ctk.CTkLabel(master, text='None', font=font)
        self.pred_time_entry.grid(row=6, column=1, padx=(10, 20), pady=(10, 10))

        # Hid layers 
        self.input_path_label = ctk.CTkLabel(master, text='Input File Path: ', font=font)
        self.input_path_label.grid(row=7, column=0, padx=(20, 10), pady=(10, 10))
        self.input_path_entry = ctk.CTkEntry(master, placeholder_text='Path to File', font=font)
        self.input_path_entry.grid(row=7, column=1, padx=(10, 20), pady=(10, 20), sticky="ew")

    def update_input_frame(self, values):
        # Values = [learning_rate, hidden_layers_size, input_size, output_size, accuracy, pred_time]

        # Hid layers 
        self.learning_rate_entry.configure(placeholder_text=str(values[0]))
        # Hid layers sizes
        self.hidden_layers_size_entry.configure(placeholder_text=str(values[1]))
        # Input size
        self.input_size_entry.configure(text=str(values[2]))
        # Output size
        self.output_size_entry.configure(text=str(values[3]))
        # Accuracy
        self.accuracy_entry.configure(text=str(values[4]))
        # Mean pred time
        self.pred_time_entry.configure(text=str(values[5]))


class NN_output_frame(ctk.CTkFrame):
    def __init__(self, master, probabilities, pred, font):
        super().__init__(master)

        rows = tuple(range(8))
        self.master.grid_columnconfigure((0, 1), weight=1)
        self.master.grid_rowconfigure(rows, weight=1)

        self.frame_title = ctk.CTkLabel(master, text='Model Prediction', font=font)
        self.frame_title.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew", columnspan=2)

        for i in range(5):
            self.probs_labels_1 = ctk.CTkLabel(master, text=f'{i}: {probabilities[i]} %', font=font)
            self.probs_labels_1.grid(row=i+1, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
            self.probs_labels_2 = ctk.CTkLabel(master, text=f'{2*i+1}: {probabilities[2*i+1]} %', font=font)
            self.probs_labels_2.grid(row=i+1, column=1, padx=(10, 10), pady=(10, 10), sticky="nsew")
        
        self.pred_label = ctk.CTkLabel(master, text=f'Prediction: {pred[0]} with {pred[1]}% confidence', font=font)
        self.pred_label.grid(row=7, column=0, padx=(10, 10), pady=(10, 10), rowspan=2, sticky="nsew", columnspan=2)
    
    def update_output_frame(self, probabilities, pred):
        for i in range(5):
            self.probs_labels_1.configure(text=f'{i}: {probabilities[i]*100} %')
            self.probs_labels_2.configure(text=f'{2*i+1}: {probabilities[2*i+1]*100} %')

        self.pred_label.configure(text=f'Prediction: {pred[0]} with {pred[1]*100}% confidence')
             

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
    global cwd, train_images, train_labels, test_images, test_labels
    cwd = os.getcwd() + "\\"

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    train_images, train_labels, test_images, test_labels = load_data("training_set/train-images-idx3-ubyte.gz", "training_set/train-labels-idx1-ubyte.gz", "test_set/t10k-images-idx3-ubyte.gz", "test_set/t10k-labels-idx1-ubyte.gz")

    Gui = Main_menu()

    Gui.mainloop()

    # self.N_network.save_params("params_file")
