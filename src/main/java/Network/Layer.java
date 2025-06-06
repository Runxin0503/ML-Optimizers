package Network;

import java.util.Arrays;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Abstract base class representing a learnable layer in a neural network.
 * <p>
 * Each layer manages its own biases, gradients, and optimizer-related state. Subclasses are responsible for defining
 * how the layer processes input and propagates gradients during training.
 * </p>
 * <p>
 * To use a layer, instantiate a subclass (e.g., DenseLayer, ConvolutionalLayer), initialize it via {@link #initialize},
 * then use {@link #calculateWeightedOutput} and {@link #updateGradient} during forward and backward passes respectively.
 * </p>
 */
public abstract class Layer {

    /** The number of Neurons in this layer */
    protected final int nodes;

    /** The bias of each neuron in this layer */
    protected final double[] bias;

    /** The bias velocity of each neuron in this layer, used in SGD with momentum */
    protected double[] biasVelocity;

    /** The bias velocity of each neuron in this layer, used in RMS-Prop */
    protected double[] biasVelocitySquared;

    /** The gradient of the bias with respect to the loss function */
    protected final double[] biasGradient;

    /** The number of times this Neural Network updated its weights and biases. */
    protected int t = 1;

    /** Creates the shell of a layer with all parameters uninitialized.
     * <br>Call {@link #initialize} with the appropriate supplier and optimizer to initialize parameters. */
    protected Layer(int nodes) {
        this.nodes = nodes;
        this.bias = new double[nodes];
        this.biasGradient = new double[nodes];
    }

    /** Initializes the parameters of this Layer */
    void initialize(Supplier<Double> initializer, Optimizer optimizer) {
        if (optimizer == Optimizer.SGD_MOMENTUM || optimizer == Optimizer.ADAM)
            this.biasVelocity = new double[nodes];
        if (optimizer == Optimizer.RMS_PROP || optimizer == Optimizer.ADAM)
            this.biasVelocitySquared = new double[nodes];
        for (int i = 0; i < nodes; i++)
            bias[i] = initializer.get();
    }

    /** Applies the learned parameters of this Layer to the given input. Returns a new array. */
    abstract double[] calculateWeightedOutput(double[] input);

    /**
     * Given the derivative array of this layer's output w.r.t the loss function (dz_dC)
     * and the previous input of this layer,
     * calculate and shift this layer's gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    abstract double[] updateGradient(double[] dz_dC, double[] x);

    /**
     * Applies this layer's gradients to the parameters of this Layer.
     * <br>Updates the respective gradient velocity vectors accordingly as well.
     */
    void applyGradient(Optimizer optimizer, double adjustedLearningRate, double momentum, double beta, double epsilon) {
        switch (optimizer) {
            case SGD -> Linalg.addInPlace(bias, Linalg.scale(-adjustedLearningRate, biasGradient));
            case SGD_MOMENTUM -> {
                Linalg.scaleInPlace(momentum, biasVelocity);
                Linalg.addInPlace(biasVelocity, Linalg.scale(1 - momentum, biasGradient));
                Linalg.addInPlace(bias, Linalg.scale(-adjustedLearningRate, biasVelocity));
            }
            case RMS_PROP -> {
                Linalg.scaleInPlace(beta, biasVelocitySquared);
                Linalg.addInPlace(biasVelocitySquared, Linalg.scale(1 - beta, Linalg.multiply(biasGradient, biasGradient)));
                IntStream.range(0, bias.length).parallel().forEach(i ->
                        bias[i] -= adjustedLearningRate * biasGradient[i] / Math.sqrt(biasVelocitySquared[i] + epsilon)
                );
            }
            case ADAM -> {
                double correctionMomentum = 1 - Math.pow(momentum, t);
                double correctionBeta = 1 - Math.pow(beta, t);
                Linalg.scaleInPlace(momentum, biasVelocity);
                Linalg.addInPlace(biasVelocity, Linalg.scale(1 - momentum, biasGradient));
                Linalg.scaleInPlace(beta, biasVelocitySquared);
                Linalg.addInPlace(biasVelocitySquared, Linalg.scale(1 - beta, Linalg.multiply(biasGradient, biasGradient)));
                double[] correctedVelocity = Linalg.scale(correctionMomentum, biasVelocity),
                        correctedVelocitySquared = Linalg.scale(correctionBeta, biasVelocitySquared);
                IntStream.range(0, bias.length).parallel().forEach(i ->
                        bias[i] -= adjustedLearningRate * correctedVelocity[i] / Math.sqrt(correctedVelocitySquared[i] + epsilon)
                );
                t++;
            }
        }
    }

    /** Clears this layer's gradient for its parameters with respect to the loss function */
    abstract void clearGradient();

    /** Returns the number of learnable parameters in this layer */
    int getNumParameters() {
        return bias.length;
    }

    @Override
    public abstract String toString();

    @Override
    public abstract Object clone();

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Layer o)) return false;
        return nodes == o.nodes &&
                Arrays.equals(bias, o.bias) &&
                Arrays.equals(biasVelocity, o.biasVelocity) &&
                Arrays.equals(biasVelocitySquared, o.biasVelocitySquared) &&
                Arrays.equals(biasGradient, o.biasGradient);
    }

    /** A helper method for subclasses of Layer to use in their {@link #toString()} methods.
     * <br>Unlike {@link Arrays#deepToString(Object[])}, this method truncates all weight numbers to 2 digits,
     * making visualization easier and less clustered. */
    static void ArraysDeepToString(double[][] array, StringBuilder sb) {
        for (int i = 0; i < array.length; i++) {
            sb.append("[");
            for (int j = 0; j < array[i].length; j++) {
                sb.append(String.format("%.2f", array[i][j]));
                if (j < array[i].length - 1)
                    sb.append(", ");
            }
            sb.append("]");
            if (i < array.length - 1)
                sb.append(",");
            sb.append("\n");
        }
    }
}
