package main;

public class NN {
    /**
     * The number of Input Neurons in this Neural Network
     */
    private final int inputNum;

    /**
     * The number of Output Neurons in this Neural Network
     */
    private final int outputNum;

    /**
     * The array of Layers in this Neural Network
     */
    private final Layer[] layers;

    /**
     * The Gradients for derivative of all Biases with respect to the loss function for all neurons in all layers of this Neural Network
     * <br>Rows: The {@link Layer} for which the weights are stored in.
     * <br>Columns: The bias deriv of a node in that layer
     */
    private double[][] biasGradient;

    /**
     * The Gradients for derivative of all Weights with respect to the loss function for all synapses between all layers of this Neural Network.
     * <br>java.Layer: The {@link Layer} for which the weights are stored in.
     * <br>Rows: The neuron {@code n} stored in that layer
     * <br>Columns: The weight deriv of a synapse pointing to {@code n}
     */
    private double[][][] weightGradient;

    /**
     * The java.Activation Function for hidden layers in this Neural Network
     */
    private final Activation hiddenAF;

    /**
     * The java.Activation Function for the output / final layer in this Neural Network
     */
    private final Activation outputAF;

    /**
     * The java.Cost Function for this Neural Network
     */
    private final Cost costFunction;

    /**
     * "Trains" the given Neural Network class using the given inputs and expected outputs.
     * <br>Uses SGD with momentum as training algorithm, requires Learning Rate and Momentum hyper-parameter.
     */
    public static void learn(NN NN, double learningRate, double momentum, double[][] testCaseInputs, double[][] testCaseOutputs) {
        //prevents other threads from calling learn on the same Neural Network
        synchronized (NN) {
            NN.clearGradient();

            Thread[] workerThreads = new Thread[testCaseInputs.length];
            for (int i = 0; i < testCaseInputs.length; i++) {
                double[] testCaseInput = testCaseInputs[i];
                double[] testCaseOutput = testCaseOutputs[i];
                workerThreads[i] = new Thread(() -> NN.backPropagate(testCaseInput, testCaseOutput));
                workerThreads[i].start();
            }

            for (Thread worker : workerThreads)
                try {
                    worker.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

            NN.applyGradient(learningRate / testCaseInputs.length, momentum);
        }
    }

    public NN(Activation hiddenAF, Activation outputAF, Cost costFunction, int... layers) {
        this.inputNum = layers[0];
        this.outputNum = layers[layers.length - 1];
        this.layers = new Layer[layers.length - 1];

        this.hiddenAF = hiddenAF;
        this.outputAF = outputAF;
        this.costFunction = costFunction;

        for (int i = 1; i < layers.length; i++)
            this.layers[i - 1] = new Layer(layers[i - 1], layers[i]);

        clearGradient();
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

        result = outputAF.calculate(result);

        assert result.length == outputNum;
        return result;
    }

    /**
     * Returns the loss of this Neural Network, or how far the expected output differs from the actual output.
     */
    public double calculateCosts(double[] input, double[] expectedOutputs) {
        double[] output = calculateOutput(input);
        double sum = 0;

        for (double v : output) assert Double.isFinite(v);

        double[] costs = costFunction.calculate(output, expectedOutputs);

        for (double v : costs) {
            sum += v;
        }

        return sum;
    }

    /**
     * Updates the {@link #weightGradient} and {@link #biasGradient} by calculating the output and
     * adding the derivative of cost function relative to each weight and bias value obtained from
     * backpropagation.
     */
    public void backPropagate(double[] input, double[] expectedOutput) {
        recursiveBackPropagation(input,expectedOutput,0);
    }

    /**
     * Given any {@code layerIndex} > 0 and its respective layer inputs, returns the derivative of the
     * input sum of all neurons in that layer.
     * <br>{@code expectedOutput} is passed down the recursive calls until output layer is reached
     *
     * @return array of da/dC or derivative of {@code layerIndex-1}'th layer's Activation Function with respective to Loss Function
     */
    private double[] recursiveBackPropagation(double[] x, double[] expectedOutput, int layerIndex) {
        if (layerIndex == layers.length - 1) {
            // x -> z -> a -> da/dC -> dz/dC -> da_-1/dC
            double[] z = layers[layerIndex].calculateWeightedOutput(x);
            double[] a = outputAF.calculate(z);

            double[] da_dC = costFunction.derivative(a,expectedOutput);
            double[] dz_dC = outputAF.derivative(z,da_dC);

            return layers[layerIndex].updateGradient(getWeightGradientLayer(layerIndex),getBiasGradientLayer(layerIndex),dz_dC,x);
        }

        // x -> z -> a -> ... -> da/dC -> dz/dC -> da_-1/dC
        double[] z = layers[layerIndex].calculateWeightedOutput(x);
        double[] a = hiddenAF.calculate(z);

        double[] da_dC = recursiveBackPropagation(a, expectedOutput, layerIndex + 1);
        double[] dz_dC = hiddenAF.derivative(z,da_dC);

        return layers[layerIndex].updateGradient(getWeightGradientLayer(layerIndex),getBiasGradientLayer(layerIndex),dz_dC,x);
    }

    /** Re-initializes the weight and bias gradients, effectively setting all contained values to 0 */
    private void clearGradient() {
        weightGradient = new double[layers.length][][];
        biasGradient = new double[layers.length][];
        for (int i = 0; i < layers.length; i++) {
            int nodes = layers[i].getNumNodes();
            weightGradient[i] = new double[nodes][(i == 0 ? inputNum : layers[i - 1].getNumNodes())];
            biasGradient[i] = new double[nodes];
        }
    }

    /**
     * Locks {@code weightGradient[layerIndex]} behind its mutex, giving access to the corresponding biasGradient as well.
     * <br>Takes up slightly more memory, but allows multiple threads to access different arrays in both weightGradient and biasGradient
     */
    private double[][] getWeightGradientLayer(int layerIndex) {
        synchronized (weightGradient[layerIndex]) {
            return weightGradient[layerIndex];
        }
    }

    /**
     * Locks {@code biasGradient[layerIndex]} behind the corresponding weightGradient array's mutex.
     * <br>Takes up slightly more memory, but allows multiple threads to access different arrays in both biasGradient and weightGradient
     */
    private double[] getBiasGradientLayer(int layerIndex) {
        synchronized (weightGradient[layerIndex]) {
            return biasGradient[layerIndex];
        }
    }

    /**
     * Applies the {@link #weightGradient} and {@link #biasGradient} derivatives to all layers in this Neural Network
     * @param adjustedLearningRate a hyper-parameter dictating how fast this Neural Network 'learn' from the given inputs
     * @param momentum a hyper-parameter dictating how much of the previous velocity to keep. [0~1]
     */
    private void applyGradient(double adjustedLearningRate, double momentum) {
        assert Double.isFinite(adjustedLearningRate);
        for (int i = 0; i < layers.length; i++) {
            for (double[] dd : weightGradient[i])
                for (double d : dd)
                    assert Double.isFinite(d);
            for (double d : biasGradient[i])
                assert Double.isFinite(d);
            layers[i].applyGradiant(weightGradient[i], biasGradient[i], adjustedLearningRate, momentum);
        }
    }
}
