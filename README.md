# Handwritten digits Identifier

## Understanding Neural Networks

Neural networks work like brains. They can *learn* and make *predictions*, just like humans. Let's break down how they work in simple terms.

### Building Blocks: Neurons and Layers

Imagine *neurons* as tiny information processors. In a neural network, these *neurons* are organized into *layers*. We have three types of *layers*:

- **Input Layer**: This *layer* receives the data to be analyzed, like images or numbers.

- **Hidden Layer(s)**: These *layers* do the processing of the data. They analyze the input and find *patterns* and *features*.

- **Output Layer**: The final *layer* gives us the network's *prediction*, like recognizing a picture or estimating a value.

### Making Predictions

To make predictions, the neural network follows these steps:

1. **Guessing**: The network starts with random guesses as it doesn't know anything because the learning process have not yet happened. These guesses are reached through mathematical operations with the inserted data. This process is called *forward propagation*

2. **Guess Checking**: It compares its guesses to the correct answers using the afforementioned process called *forward propagation*. This is a way to measure how wrong or right the neural network is.

3. **Learning from Mistakes**: By measuring the difference between its guesses and the correct answers (called *loss function*), the network figures out what went wrong. Here we assume that we already know the correct answer (*labeled data*) for the given input which is called *supervised learning*.

4. **Correcting Course**: It adjusts its guesses based on the mistakes it has made using an optimization algorithm called *gradient descent*. This makes the network to get closer to the essence of what the data is. In short terms, the *gradient descend* algorithm undoes the prediction to figure how the data shoud've been treated and correct the neural network's parameters accordingly.  

5. **Repeating and Improving**: All these steps are repeated many times until the network becomes good at making accurate predictions. We have many ways to evaluate how good a neural network performs but accuracy is the most intuitive property.

### Learning with Labels: Supervised Learning

Neural networks often learn through supervised learning. It's like having a teacher who tells you if your answers are right or wrong so you can improve for next the time. Here's how it works:

1. **Teacher's Instructions**: The network is given a set of *labeled* examples. For instance, pictures of cats and dogs labeled as "cat" or "dog."

2. **Observation and Imitation**: The network studies the *labeled* examples and tries to understand the *patterns* on its own. It tries to imitate the teacher.

3. **Teacher's Feedback**: The network compares its *predictions* with the correct labels and learns from the *mistakes*. It knows if it got a prediction right or wrong.

4. **Getting Better**: By repeating this process with many examples, the network improves its ability to make accurate predictions.

Neural networks are powerful problem solvers that can learn from data, making them useful in many fields like image recognition, speech understanding, and even self-driving cars. Lets take a look into the digit identifier built in this proyect.

## The Maths behind Neural Networks

### Activation Function

The activation function introduces non-linearity into the neural network, allowing it to learn complex relationships. The purpose of the activation function is to determine how relevant is the information contained within a neuron and transfer it accordingly. Common activation functions include:

- Sigmoid Function: $\sigma(x) = \frac{1}{1 + e^{-x}}$

- Rectified Linear Unit (ReLU): $f(x) = \max(0, x)$

- Hyperbolic tangent (tanh): $f(x) = \tanh(x)$

### Loss Function

The loss function measures the difference between the predicted output and the true target value. Common loss functions for different types of problems are:

- Mean Squared Error (MSE): $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y_i})^2$

- Cross-Entropy Loss (for classification): $\text{CrossEntropyLoss} = -\sum_{i=1}^{n} y_i \log(\hat{y_i})$

### Optimization Algorithm: Gradient Descent

Gradient descent is an optimization algorithm used to minimize the loss function and update the weights and biases in the neural network. The steps involved in gradient descent are:

1. Calculate the gradients of the loss function with respect to the network's parameters using backpropagation. In other words, we are calculation how much every value affects the result.
    - $\frac{\partial C_{0}}{\partial a_i} = \frac{\partial C_0}{\partial a_{i+1}} \frac{\partial a_{i+1}}{\partial z_{i+1}} \frac{\partial z_{i+1}}{\partial a_i}$
    - $\frac{\partial C_0}{\partial \omega_i} = \frac{\partial C_0}{\partial a_i} \frac{\partial a_i}{\partial z_i} \frac{\partial z_i}{\partial \omega_i}$
    - $\frac{\partial C_0}{\partial b_i} = \frac{\partial C_0}{\partial a_i} \frac{\partial a_i}{\partial z_i} \frac{\partial z_i}{\partial b_i}$

     (Where $C_0$ is the loss funtion, $a_i$ is the activation of each neuron of a layer, $\omega_i$ are the weights of each neuron of a layer and $b_i$ is the bias of each neuron of a layer)

2. Update the parameters by subtracting a fraction of the gradients multiplied by the learning rate.
    - $\omega'_i = \omega_i - \alpha \frac{\partial C_0}{\partial \omega_i}$
    - $b'_i = b_i - \alpha \frac{\partial C_0}{\partial b_i}$
     (Where $\alpha$ is the learing rate and $\omega'_i, b'_i$ are the updated parameters)

3. Repeat these steps iteratively through the training set. If the accuracy converges (or the performance parameter of choice), it is best to stop the training process becuase no better performance would be achieved so it would be a lost of time and computation power to continue trying to improve the neural network. Also the training porcess could be stopped if a chosen amount of iteration is reached though it is rarely the case where this threshold is useful. 

### Digit Identifier

As a reference, the sample network has four layers. Two for the input and output and two hidden layers. The input has 784 neurons and each of the hidden layers plus the output have 10 neurons. The input is more or less forced as the images that are to be analyzed are 28 by 28 pixels (784 in total) and same thing happens to the output layer as we need 10 neurons, one for each digit. On the other hand, the hidden layers are arbitrarily chosen and other geometries and sizes may yield better results. Here is how the neural network looks:

<p align="center">
  <img src="[images\NN_digit_structure.png](https://github.com/amprod18/NN_digit_detector/blob/main/images/NN_digit_structure.png)" alt="NN_digit_structure" height="500 px" length="500 px">
</p>

