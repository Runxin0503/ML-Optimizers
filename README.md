# ML-Optimizers

A Java-based neural network framework built from scratch, designed to experiment with and compare different
gradient-based optimization algorithms. It supports a modular and flexible architecture for constructing, training, and
evaluating both dense and convolutional neural networks.

---

## 🚀 Features

- **Custom Neural Network Framework**:
    - Dense and Convolutional layers
    - Customizable architecture via builder pattern
    - Softmax temperature control (ideal for RL exploration)
    - Threaded batch training for performance

- **Built-in Optimizers**:
    - SGD (Stochastic Gradient Descent)
    - SGD with Momentum
    - RMSProp
    - Adam

- **Trainable on Diverse Tasks**:
    - Logic Gates: NOT, OR, AND, XOR
    - Linear Regression with randomly generated functions
    - MNIST digit recognition
    - Iris flower classification

- **Fully Extensible**:
    - Plug in custom layers, activation functions, and loss functions
    - Use in supervised learning or extend to reinforcement learning

---

## Network Class (NN)

The NN class represents a customizable neural network. It supports both dense and convolutional architectures. The
network handles forward propagation, backpropagation, cost calculation, and gradient-based optimization using a
configurable optimizer such as SGD, RMSProp, or Adam. It also supports multi-threaded training and can be used for both
supervised learning and reinforcement learning scenarios.

You can configure the network using the NetworkBuilder class to define the architecture, activation functions, and
optimizer.

## 🧠 Example Usage

To create a neural network, you can use the builder pattern provided in the NetworkBuilder class:

```java
NN neuralNetwork = new NN.NetworkBuilder()
        .setInputNum(784)  // Input layer size for MNIST
        .addDenseLayer(128)  // Add a dense hidden layer with 128 neurons
        .addDenseLayer(10)   // Output layer with 10 neurons (for classification)
        .setHiddenAF(Activation.relu)  // Set the activation function for hidden layers
        .setOutputAF(Activation.softmax)  // Set the activation function for output layer
        .setCostFunction(Cost.crossEntropy)  // Use cross-entropy as cost function
        .setOptimizer(Optimizer.ADAM)  // Use Adam optimizer
        .build();  // Build the neural network
```

## Training the Network

Once the network is built, you can train it using the learn method:

```java
NN.learn(neuralNetwork, learningRate, momentum, beta, epsilon, trainingInputs, trainingOutputs);
```

This will optimize the network weights using the chosen optimizer and hyperparameters.

## Backpropagation for a Single Output

For reinforcement learning scenarios, you can train for a single output:

```java
NN.learnSingleOutput(neuralNetwork, learningRate, momentum, beta, epsilon, input, outputIndex, expectedOutput);
```

## Test Suite

Includes a variety of tests to benchmark and validate performance:

- **Logic Gates**:

  NOT, OR, AND, XOR — The model learns logical operations from binary input-output pairs.


- **Linear Function Regression**

  Learns the slope and intercept of randomly generated lines with no hidden layers.


- **Iris Dataset**

  Classifies flower species from petal/sepal data using a simple feedforward architecture.


- **MNIST Dataset**

  Handwritten digit classification using convolutional and dense layers.

## Maven Dependencies

Here is the dependency you can add to your pom.xml file for using this package in your own project:

```xml

<dependency>
    <groupId>io.github.runxin0503</groupId>
    <artifactId>ml-optimizers</artifactId>
    <version>1.0.0</version>
</dependency>
```

## Contribution

If you want to contribute to this project, feel free to fork the repository and submit a pull request. Make sure to
include tests for any new functionality and adhere to the existing coding style.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
