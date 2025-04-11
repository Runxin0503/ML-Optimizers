package Network;

import enums.Activation;
import enums.Cost;
import enums.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * TODO write documentation
 */
public class NN {
    /**
     * The number of Input Neurons in this Neural Network
     */ //made public for testing purposes, was private
    public final int inputNum;

    /**
     * The number of Output Neurons in this Neural Network
     */ //made public for testing purposes, was private
    public final int outputNum;

    /**
     * The array of Layers in this Neural Network
     */ //made public for testing purposes, was private
    public final Layer[] layers;

    /**
     * The value for encouraging exploration in softmax (discrete) actions in a Reinforcement Learning environment
     */
    private double temperature;

    /**
     * The Activation Function for hidden layers in this Neural Network
     */
    private final Activation hiddenAF;

    /**
     * The Activation Function for the output / final layer in this Neural Network
     */
    private final Activation outputAF;

    /**
     * The Cost Function for this Neural Network
     */
    private final Cost costFunction;

    /**
     * The Optimizer class used for this Neural Network's backpropagation and training process.
     */
    private final Optimizer optimizer;

    /**
     * "Trains" the given Neural Network class using the given batches of input and expected output.
     * <br>Depending on the {@link Optimizer}, this function requires different parameters:
     * <br>-{@link Optimizer#SGD}: Learning Rate
     * <br>-{@link Optimizer#SGD_MOMENTUM}: Learning Rate, momentum
     * <br>-{@link Optimizer#RMS_PROP}: Learning Rate, beta, and epsilon
     * <br>-{@link Optimizer#ADAM}: Learning Rate, momentum, beta, and epsilon
     * @param learningRate a hyper-parameter dictating how fast this Neural Network 'learn' from the given inputs
     * @param momentum a hyper-parameter dictating how much of the previous SGD velocity to keep. [0~1]
     * @param beta a hyper-parameter dictating how much of the previous RMS-Prop velocity to keep. [0~1]
     * @param epsilon a hyper-parameter that's typically very small to avoid divide by zero errors
     */
    public static void learn(NN NN, double learningRate, double momentum, double beta, double epsilon, double[][] testCaseInputs, double[][] testCaseOutputs) {
        assert testCaseInputs.length == testCaseOutputs.length;
        for (int i = 0; i < testCaseInputs.length; ++i)
            assert testCaseInputs[i].length == NN.inputNum && testCaseOutputs[i].length == NN.outputNum;
        //prevents other threads from calling learn on the same Neural Network
        synchronized (NN) {
            NN.clearGradient();

            Thread[] workerThreads = new Thread[testCaseInputs.length];
            for (int i = 0; i < testCaseInputs.length; i++) {
                double[] testCaseInput = testCaseInputs[i];
                double[] testCaseOutput = testCaseOutputs[i];
                workerThreads[i] = new Thread(null, () -> NN.backPropagate(testCaseInput, testCaseOutput), "WorkerThread");
                workerThreads[i].start();
            }

            for (Thread worker : workerThreads)
                try {
                    worker.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

            NN.applyGradient(NN.optimizer, learningRate / testCaseInputs.length, momentum, beta, epsilon);
        }
    }

    /**
     * Runs the optimizer in this Neural Network class with the given input and a single expected output.
     * <br>Unlike {@link #learn}, this function does backpropagation on a single output element instead of an entire output vector.
     * <br>Depending on the {@link Optimizer}, this function requires different parameters:
     * <br>-{@link Optimizer#SGD}: Learning Rate
     * <br>-{@link Optimizer#SGD_MOMENTUM}: Learning Rate, momentum
     * <br>-{@link Optimizer#RMS_PROP}: Learning Rate, beta, and epsilon
     * <br>-{@link Optimizer#ADAM}: Learning Rate, momentum, beta, and epsilon     * @param learningRate a hyper-parameter dictating how fast this Neural Network 'learn' from the given inputs
     * @param momentum a hyper-parameter dictating how much of the previous SGD velocity to keep. [0~1]
     * @param beta a hyper-parameter dictating how much of the previous RMS-Prop velocity to keep. [0~1]
     * @param epsilon a hyper-parameter that's typically very small to avoid divide by zero errors
     * @param outputIndex the index of the element to optimize
     */
    public static void learnSingleOutput(NN NN, double learningRate, double momentum, double beta, double epsilon, double[] input, int outputIndex, double expectedOutput) {
        synchronized (NN) {
            double[] output = NN.calculateOutput(input);
            assert 0 <= outputIndex && outputIndex < NN.outputNum;
            output[outputIndex] = expectedOutput;

            NN.clearGradient();
            NN.backPropagate(input, output);
            NN.applyGradient(NN.optimizer, learningRate, momentum, beta, epsilon);
        }
    }

    private NN(Optimizer optimizer, int inputNum, int outputNum, double temperature, Activation hiddenAF, Activation outputAF, Cost costFunction, Layer[] layers) {
        this.inputNum = inputNum;
        this.outputNum = outputNum;
        this.layers = layers;
        this.optimizer = optimizer;

        this.temperature = temperature;
        this.hiddenAF = hiddenAF;
        this.outputAF = outputAF;
        this.costFunction = costFunction;

        clearGradient();
    }

    /** Sets the temperature (exploration) value of this Neural Network.
     * <br>Only affects the output when {@link #outputAF} is {@link Activation#softmax} */
    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    /**
     * Applies the weights and biases of this Neural Network to transform the {@code input} array to an
     * {@code output} array of predictions
     */
    public double[] calculateOutput(double[] input) {
        assert input.length == inputNum;

        double[] result = layers[0].calculateWeightedOutput(input);
        for (int i = 1; i < layers.length; i++) {
            result = layers[i].calculateWeightedOutput(hiddenAF.calculate(result));
        }

        //exploration vs exploitation. Apply temperature in softmax function for RL algorithms
        if (outputAF == Activation.softmax)
            for (int i = 0; i < result.length; i++)
                result[i] /= temperature;

        result = outputAF.calculate(result);

        assert result.length == outputNum;
        return result;
    }

    /**
     * Returns the loss of this Neural Network, or how far the expected output differs from the actual output.
     */
    public double calculateCost(double[] input, double[] expectedOutput) {
        double[] output = calculateOutput(input);
        double sum = 0;

        for (double v : output) assert Double.isFinite(v);

        double[] costs = costFunction.calculate(output, expectedOutput);

        for (double v : costs) {
            sum += v;
        }

        return sum;
    }

    /**
     * Populates each layer's gradient parameters by calculating the output and
     * adding the derivative of cost function relative to each weight and bias value obtained from
     * backpropagation.
     */
    public void backPropagate(double[] input, double[] expectedOutput) {
        //z = immediate output of every layer (right before activation function)
        //x = immediate input of every layer (either is input or is right after activation function)
        double[][] zs = new double[layers.length][];
        double[][] xs = new double[layers.length][];
        xs[0] = input;
        for (int i = 0; i < layers.length - 1; i++) {
            zs[i] = layers[i].calculateWeightedOutput(xs[i]);
            xs[i + 1] = hiddenAF.calculate(zs[i]);
        }
        zs[layers.length - 1] = layers[layers.length - 1].calculateWeightedOutput(xs[layers.length - 1]);
        if (outputAF == Activation.softmax)
            for (int i = 0; i < zs[layers.length - 1].length; i++)
                zs[layers.length - 1][i] /= temperature;

        double[] output = outputAF.calculate(zs[layers.length - 1]);
        if (outputAF == Activation.softmax)
            for (int i = 0; i < output.length; i++)
                output[i] /= temperature;

        double[] outputLayer_dz_dC = outputAF.derivative(zs[layers.length - 1], costFunction.derivative(output, expectedOutput));
        double[] nextLayer_da_dC;
        nextLayer_da_dC = layers[layers.length - 1].updateGradient(outputLayer_dz_dC, xs[layers.length - 1]);

        for (int i = layers.length - 2; i >= 0; i--) {
            double[] dz_dC = hiddenAF.derivative(zs[i], nextLayer_da_dC);
            nextLayer_da_dC = layers[i].updateGradient(dz_dC, xs[i]);
        }
    }

    /** Re-initializes the weight and bias gradients, effectively setting all contained values to 0 */
    private void clearGradient() {
        for (Layer layer : layers) layer.clearGradient();
    }

    /**
     * Applies the gradients of each layer in this Neural Network to itself
     */
    private void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        assert Double.isFinite(adjustedLearningRate);
        for (Layer layer : layers)
            layer.applyGradient(optimizer, adjustedLearningRate, momentum, beta, epsilon);
    }

    @Override
    public String toString() {
        int totalParameters = 0;
        for (Layer layer : layers) totalParameters += layer.getNumParameters();

        StringBuilder sb = new StringBuilder();
        sb.append("Network with ").append(totalParameters).append(" parameters\n");
        for (int i = 0; i < layers.length; i++) {
            sb.append("Layer ").append(i).append("\n");
            sb.append(layers[i].toString());
        }
        return sb.toString();
    }

    @Override
    public Object clone() {
        Layer[] newLayers = new Layer[layers.length];
        for (int i = 0; i < layers.length; i++) newLayers[i] = (Layer) layers[i].clone();
        return new NN(optimizer, inputNum, outputNum, temperature, hiddenAF, outputAF, costFunction, newLayers);
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof NN o)) return false;
        return inputNum == o.inputNum && outputNum == o.outputNum &&
                temperature == o.temperature &&
                hiddenAF == o.hiddenAF && outputAF == o.outputAF &&
                costFunction == o.costFunction &&
                Arrays.equals(layers, o.layers);
    }

    public static class NetworkBuilder {
        private int inputNum = -1, outputNum = -1;
        private Activation hiddenAF = null;
        private Activation outputAF = null;
        private Cost costFunction = null;
        private double temperature = 1;
        private Optimizer optimizer = Optimizer.ADAM;
        private final ArrayList<Layer> layers = new ArrayList<>();

        //TODO add a "addLayer" method for general Layer
        public NetworkBuilder setInputNum(int inputNum) {
            this.inputNum = inputNum;
            if (!layers.isEmpty()) {
                Layer newLayer;
                if (layers.getFirst() instanceof DenseLayer d)
                    newLayer = new DenseLayer(inputNum, d.nodes);
                else if (layers.getFirst() instanceof ConvolutionalLayer)
                    throw new UnsupportedOperationException(
                            "Attempted to declare NN.NetworkBuilder.setInputNum after adding Convolutional layers.");
                else throw new RuntimeException();
                layers.set(0, newLayer);
            }
            return this;
        }

        public NetworkBuilder addDenseLayer(int nodes) {
            if (layers.isEmpty()) layers.add(new DenseLayer(inputNum, nodes));
            else layers.add(new DenseLayer(layers.getLast().nodes, nodes));
            outputNum = layers.getLast().nodes;
            return this;
        }

        public NetworkBuilder addConvolutionalLayer(int inputWidth, int inputHeight, int inputLength,
                                                    int kernelWidth, int kernelHeight, int numKernels,
                                                    int strideWidth, int strideHeight) {
            return addConvolutionalLayer(inputWidth, inputHeight, inputLength, kernelWidth, kernelHeight, numKernels, strideWidth, strideHeight, false);
        }

        public NetworkBuilder addConvolutionalLayer(int inputWidth, int inputHeight, int inputLength,
                                                    int kernelWidth, int kernelHeight, int numKernels,
                                                    int strideWidth, int strideHeight, boolean padding) {
            assert (layers.isEmpty() ? inputNum : layers.getLast().nodes) == inputWidth * inputHeight * inputLength;
            layers.add(new ConvolutionalLayer(inputWidth, inputHeight, inputLength, kernelWidth, kernelHeight, numKernels, strideWidth, strideHeight, padding));
            outputNum = layers.getLast().nodes;
            return this;
        }

        public NetworkBuilder setHiddenAF(Activation hiddenAF) {
            this.hiddenAF = hiddenAF;
            return this;
        }

        public NetworkBuilder setOutputAF(Activation outputAF) {
            this.outputAF = outputAF;
            return this;
        }

        public NetworkBuilder setCostFunction(Cost costFunction) {
            this.costFunction = costFunction;
            return this;
        }

        public NetworkBuilder setTemperature(double temperature) {
            this.temperature = temperature;
            return this;
        }

        public NetworkBuilder setOptimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public NN build() throws MissingInformationException {
            if (inputNum == -1 || outputNum == -1 || hiddenAF == null || outputAF == null || costFunction == null || layers.isEmpty() || optimizer == null)
                throw new MissingInformationException();
            for (Layer layer : layers)
                layer.initialize(Activation.getInitializer(hiddenAF, inputNum, outputNum), optimizer);
            return new NN(optimizer, inputNum, outputNum, temperature, hiddenAF, outputAF, costFunction, layers.toArray(new Layer[0]));
        }
    }

    public static class MissingInformationException extends RuntimeException {
        @Override
        public String getMessage() {
            return "Missing argument when creating new NN object";
        }
    }
}
