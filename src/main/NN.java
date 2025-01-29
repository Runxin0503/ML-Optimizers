package main;

public class NN {
    /**
     * The number of Input Neurons in this Neural Network
     */ //made public for testing purposes
    public final int inputNum;

    /**
     * The number of Output Neurons in this Neural Network
     */ //made public for testing purposes
    public final int outputNum;

    /**
     * The array of Layers in this Neural Network
     */ //made public for testing purposes
    public final Layer[] layers;

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
     * <br>Uses RMS-Prop as training algorithm, requires Learning Rate, beta, and epsilon hyper-parameter.
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

            NN.applyGradient(learningRate / testCaseInputs.length, momentum, beta, epsilon);
        }
    }

    public NN(Activation hiddenAF, Activation outputAF, Cost costFunction, int... layers) {
        this.inputNum = layers[0];
        this.outputNum = layers[layers.length - 1];
        this.layers = new DenseLayer[layers.length - 1];

        this.hiddenAF = hiddenAF;
        this.outputAF = outputAF;
        this.costFunction = costFunction;

        for (int i = 1; i < layers.length - 1; i++) {
            this.layers[i - 1] = new DenseLayer(layers[i - 1], layers[i], Activation.getInitializer(hiddenAF, inputNum, outputNum));
        }

        this.layers[layers.length - 2] = new DenseLayer(layers[layers.length - 2], layers[layers.length - 1], Activation.getInitializer(outputAF, inputNum, outputNum));

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
     * Populates each layer's gradient parameters by calculating the output and
     * adding the derivative of cost function relative to each weight and bias value obtained from
     * backpropagation.
     */
    public void backPropagate(double[] input, double[] expectedOutput) {
        recursiveBackPropagation(input, expectedOutput, 0);
        //todo unravel recursion
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

            double[] da_dC = costFunction.derivative(a, expectedOutput);
            double[] dz_dC = outputAF.derivative(z, da_dC);

            return layers[layerIndex].updateGradient(dz_dC, x);
        }

        // x -> z -> a -> ... -> da/dC -> dz/dC -> da_-1/dC
        double[] z = layers[layerIndex].calculateWeightedOutput(x);
        double[] a = hiddenAF.calculate(z);

        double[] da_dC = recursiveBackPropagation(a, expectedOutput, layerIndex + 1);
        double[] dz_dC = hiddenAF.derivative(z, da_dC);

        return layers[layerIndex].updateGradient(dz_dC, x);
    }

    /** Re-initializes the weight and bias gradients, effectively setting all contained values to 0 */
    private void clearGradient() {
        for(Layer layer : layers) layer.clearGradient();
    }

    /**
     * Applies the gradients of each layer in this Neural Network to itself
     */
    private void applyGradient(double adjustedLearningRate, double momentum, double beta, double epsilon) {
        assert Double.isFinite(adjustedLearningRate);
        for (Layer layer : layers)
            layer.applyGradiant(adjustedLearningRate, momentum, beta, epsilon);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < layers.length; i++) {
            sb.append("Layer ").append(i).append("\n");
            sb.append(layers[i].toString());
        }
        return sb.toString();
    }
}
