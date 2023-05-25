import time
import gzip
import os
import numpy as np
import custom_NN
import customtkinter as ctk
from PIL import Image


class Main_menu(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0
        self.delta = 0.05

        self.overrideredirect(True) # Used for fullscreen later on
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.title("Neural Network Digit Identifier")
        self.geometry(f"{screen_width}x{screen_height}")
        self.geometry("+0+0")
        self.main_frame = ctk.CTkTabview(self, width=screen_width, height=screen_height)
        self.main_frame.pack(padx=(20, 0), pady=(20, 0))
        self.main_frame.add("Loading Screen")
        self.main_frame.add("Main Menu")
        self.exit_button_image = ctk.CTkImage(light_image=Image.open("images/exit_button.png"), size=(20, 20))
        self.exit_button = ctk.CTkButton(self, text="", width=20, height=20, image=self.exit_button_image, corner_radius=5, fg_color="#F57A91", hover_color='#FF002C')
        self.exit_button.configure(command=self.exit_app)
        self.exit_button.place(x=screen_width-40, y=10)
        
        
        self.logo = ctk.CTkImage(light_image=Image.open("images/neural_network_icon_lth.png"), dark_image=Image.open("images/neural_network_icon_dth.png"), size=(250, 250))
        image_width, image_height = 250, 250
        self.start_logo = ctk.CTkLabel(self.main_frame.tab("Loading Screen"), image=self.logo, text="")
        x = (screen_width - image_width) // 2
        y = (screen_height - image_height) // 2
        f = lambda event: self.initial_progress_bar(event, x, y)
        self.bind('<Button-1>', f)
        self.bind('<Key>', f)
        self.start_logo.place(x=x, y=y)

        self.progressbar = ctk.CTkProgressBar(self.main_frame.tab("Loading Screen"), width=image_width, mode='determinate', determinate_speed=0.2)
        self.progressbar.set(0)
        y = y + image_height + 30
        
        self.start_text = ctk.CTkLabel(self.main_frame.tab("Loading Screen"), width=image_width, height=50, text="Press Anywhere to Continue", fg_color='transparent')
        self.start_text.place(x=x, y=y)

    def initial_progress_bar(self, event, x, y):
        self.unbind_all(('<Button-1>', '<Key>'))
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

        self.main_menu()
    
    def main_menu(self):
        self.update_idletasks()
        # self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        # self.grid_rowconfigure((0, 1, 2), weight=1)

        self.frame_1 = ctk.CTkFrame(self.main_frame.tab("Main Menu"), fg_color="transparent")
        self.frame_2 = ctk.CTkFrame(self.main_frame.tab("Main Menu"), fg_color="transparent")
        self.frame_3 = ctk.CTkFrame(self.main_frame.tab("Main Menu"), fg_color="transparent")
        self.frame_1.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.frame_2.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.frame_3.grid(row=0, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.frame_1_1 = ctk.CTkFrame(self.frame_1, fg_color="transparent")
        self.frame_1_2 = ctk.CTkFrame(self.frame_1, fg_color="transparent")
        self.frame_1_3 = ctk.CTkFrame(self.frame_1, fg_color="transparent")
        self.frame_1_1.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.frame_1_2.grid(row=1, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.frame_1_3.grid(row=2, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")

        self.frame_3_1 = ctk.CTkFrame(self.frame_3, fg_color="transparent")
        self.frame_3_2 = ctk.CTkFrame(self.frame_3, fg_color="transparent")
        self.frame_3_1.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.frame_3_2.grid(row=1, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")

        self.load_model_button = ctk.CTkButton(self.frame_1_1, text="Load Model")
        self.train_model_button = ctk.CTkButton(self.frame_1_1, text="Train Model")
        self.load_model_button.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="")
        self.train_model_button.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="")

        self.predict_button = ctk.CTkButton(self.frame_1_3, text="Predict")
        self.predict_button.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="")

        self.input_image = ctk.CTkLabel(self.frame_3_1, text="Input image will appear here")
        self.input_image.grid(row=0, column=0, padx=(20, 20), pady=(20, 0), sticky="nsew")

        self.main_frame.set("Main Menu")
    
    def exit_app(self):
        time.sleep(0.5)
        self.destroy()


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
