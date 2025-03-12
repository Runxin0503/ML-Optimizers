package main;

import java.util.Arrays;
import java.util.function.Supplier;

public abstract class Layer {


    /** The number of Neurons in this layer */
    //made public for testing purposes, was protected
    public final int nodes;

    /** The bias of each neuron in this layer */
    //made public for testing purposes, was protected
    public final double[] bias;

    /** The bias velocity of each neuron in this layer, used in SGD with momentum */
    //made public for testing purposes, was protected
    public final double[] biasVelocity;

    /** The bias velocity of each neuron in this layer, used in RMS-Prop */
    //made public for testing purposes, was protected
    public final double[] biasVelocitySquared;

    /** The gradient of the bias with respect to the loss function */
    protected final double[] biasGradient;

    /** The number of times this Neural Network updated its weights and biases. */
    //made public for testing purposes, was protected
    public int t = 1;

    public Layer(int nodes) {
        this.nodes = nodes;
        this.bias = new double[nodes];
        this.biasVelocity = new double[nodes];
        this.biasVelocitySquared = new double[nodes];
        this.biasGradient = new double[nodes];

    }

    /** Initializes the parameters of this Layer */
    public void initialize(Supplier<Double> initializer){
        for (int i = 0; i < nodes; i++)
            bias[i] = initializer.get();
    }

    /** Passes the input through this Layer and returns a new array of numbers. */
    public abstract double[] calculateWeightedOutput(double[] input);

    /**
     * Given the derivative array of the latest input sum (dz_dC),
     * calculate and shift this layer's gradients.
     * @return da_dC where a is the activation function of the layer before this one
     */
    public abstract double[] updateGradient(double[] dz_dC, double[] x);

    /**
     * Applies this layer's gradients to the parameters of this Layer.
     * <br>Updates the respective gradient velocity vectors accordingly as well.
     */
    public abstract void applyGradient(double adjustedLearningRate, double momentum, double beta, double epsilon);

    /** Clears this layer's gradient for its parameters with respect to the loss function */
    public abstract void clearGradient();

    /** Returns the number of learnable parameters in this layer */
    public int getNumParameters() {
        return bias.length;
    }

    @Override
    public abstract String toString();

    @Override
    public abstract Object clone();

    @Override
    public boolean equals(Object obj){
        if(!(obj instanceof Layer o)) return false;
        return nodes == o.nodes &&
                Arrays.equals(bias, o.bias) &&
                Arrays.equals(biasVelocity, o.biasVelocity) &&
                Arrays.equals(biasVelocitySquared, o.biasVelocitySquared) &&
                Arrays.equals(biasGradient, o.biasGradient);
    }

    /** A helper method for subclasses of Layer to use in their {@link #toString()} methods.
     * <br>Unlike {@link Arrays#deepToString(Object[])}, this method truncates all weight numbers to 2 digits,
     * making visualization easier and less clustered. */
    static void ArraysDeepToString(double[][] array,StringBuilder sb) {
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
