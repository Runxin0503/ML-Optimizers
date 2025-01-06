# ML-Optimizers

Contains a collection of releases for implementations of the following optimizers:
- SGD (Stochastic Gradient Descent)
- SGD with Momentum
- RMSP / RMS-Prop (Root Mean Squared Propagation)
- Adam (Adaptive Moment Estimation)

The hyper-parameters of each optimizer were then tuned to near-perfect performance on each test in the test suite.

The test suite contains:
- Logic gates (NOT, OR, AND, XOR), where the model trains to behave like a logic gate when given binary inputs
- Linear Functions, where the model has no hidden neurons and 1 input & output neuron and learns the weight and bias of a randomly generated linear function
- MNIST dataset, where the model has to recognize hand-written numbers
- Iris dataset, where the model trains to separate plants into different species by their physical qualities

The purpose of this repo was mainly for me to learn the benefits of each optimizer and understand conceptually how the optimizers work beyond the surface level.
