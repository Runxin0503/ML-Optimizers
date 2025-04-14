package Network;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Enum representing common activation functions and their derivatives used in neural networks.
 * <p>
 * Each activation constant includes:
 * <ul>
 *     <li>The forward pass function {@code f(z)}</li>
 *     <li>The backward pass derivative function {@code f'(z) * ∂a/∂C}</li>
 * </ul>
 * This enum also provides recommended weight initialization strategies (He or Xavier) based on the selected activation function.
 */
public enum Activation {
    /**
     * No activation applied. Acts as a passthrough (identity function).
     */
    none(input -> {
        double[] output = new double[input.length];
        System.arraycopy(input, 0, output, 0, input.length);
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        //input will always be 1 with respect to output
        System.arraycopy(gradient, 0, output, 0, input.length);
        return output;
    }),

    /**
     * Rectified Linear Unit: f(x) = max(0, x)
     */
    ReLU(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (input[i] > 0 ? input[i] : 0);
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = gradient[i] * (input[i] > 0 ? 1.0 : 0);
        return output;
    }),

    /**
     * Sigmoid activation: f(x) = 1 / (1 + e^-x)
     */
    sigmoid(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = 1 / (1 + Math.exp(-input[i]));
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double a = 1 / (1 + Math.exp(-input[i]));
            output[i] = gradient[i] * a * (1 - a);
        }
        return output;
    }),

    /**
     * Hyperbolic Tangent: f(x) = tanh(x)
     */
    tanh(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
            output[i] = Math.tanh(input[i]);
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double tanhValue = Math.tanh(input[i]);
            output[i] = gradient[i] * (1 - tanhValue * tanhValue);
        }
        return output;
    }),

    /**
     * Leaky ReLU: f(x) = x if x > 0 else 0.1 * x
     */
    LeakyReLU(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = input[i] > 0 ? input[i] : 0.1 * input[i];
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = gradient[i] * (input[i] > 0 ? 1.0 : 0.1);
        return output;
    }),

    /**
     * Softmax activation for converting logits into probabilities.
     * Numerically stable using max-shift trick.
     */
    softmax(input -> {
        double[] output = new double[input.length];
        double latestInputSum = 0, max = Double.MIN_VALUE;
        for (double num : input) max = Math.max(max, num);
        for (double num : input) latestInputSum += Math.exp(num - max);
        for (int i = 0; i < input.length; i++) output[i] = Math.exp(input[i] - max) / latestInputSum;
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        double[] softmaxOutput = new double[input.length];
        double latestInputSum = 0, max = Double.MIN_VALUE;
        for (double num : input) max = Math.max(max, num);
        for (double num : input) latestInputSum += Math.exp(num - max);
        for (int i = 0; i < input.length; i++) softmaxOutput[i] = Math.exp(input[i] - max) / latestInputSum;

        // Compute the gradient using the vectorized form
        double dotProduct = 0.0;
        for (int i = 0; i < softmaxOutput.length; i++) {
            dotProduct += softmaxOutput[i] * gradient[i];
        }

        for (int i = 0; i < softmaxOutput.length; i++) {
            output[i] = softmaxOutput[i] * (gradient[i] - dotProduct);
        }

        return output;
    });

    private static final Random RANDOM = new Random();
    private static final BiFunction<Integer, Integer, Double> HE_Initialization = (inputSize, outputSize) -> RANDOM.nextGaussian(0, Math.sqrt(2.0 / (inputSize + outputSize)));
    private static final BiFunction<Integer, Integer, Double> XAVIER_Initialization = (inputSize, outputSize) -> RANDOM.nextGaussian(0, Math.sqrt(1 / Math.sqrt(inputSize + outputSize)));

    private final Function<double[], double[]> function;
    private final BiFunction<double[], double[], double[]> derivativeFunction;

    Activation(Function<double[], double[]> function, BiFunction<double[], double[], double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /**
     * Applies the activation function to the input array.
     * @param input Array of pre-activation values (z)
     * @return Array of post-activation values (a)
     * @throws AssertionError if input or output contains non-finite values
     */
    double[] calculate(double[] input) {
        for (double v : input)
            assert Double.isFinite(v) : "Attempted to input invalid values into Activation Function " + Arrays.toString(input);
        double[] output = this.function.apply(input);
        for (double v : output)
            assert Double.isFinite(v) : "Activation Function returning invalid values " + Arrays.toString(input) + "\n" + Arrays.toString(output);
        return output;
    }

    /**
     * Computes the derivative of the activation with respect to cost.
     * <p>
     * This is used in backpropagation: {@code dz/dC = da/dC * f'(z)}
     * @param z         Pre-activation values from the forward pass
     * @param da_dC     Gradient of activation output with respect to cost
     * @return Gradient of pre-activation with respect to cost
     */
    double[] derivative(double[] z, double[] da_dC) {
        for (double v : da_dC)
            assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Activation Function " + Arrays.toString(z) + "  " + Arrays.toString(da_dC);
        double[] newGradient = this.derivativeFunction.apply(z, da_dC);
        for (double v : newGradient)
            assert Double.isFinite(v) : "Deriv of Activation Function returning invalid values " + Arrays.toString(z) + "  " + Arrays.toString(da_dC) + "\n" + Arrays.toString(newGradient);
        return newGradient;
    }

    /**
     * Returns the recommended weight initialization function for a given activation type.
     * <ul>
     *     <li>Uses He initialization for ReLU and LeakyReLU</li>
     *     <li>Uses Xavier initialization otherwise</li>
     * </ul>
     * @param AF Activation function
     * @param inputNum   Number of input units
     * @param outputNum  Number of output units
     * @return Supplier that generates properly scaled initial weights
     */
    static Supplier<Double> getInitializer(Activation AF, int inputNum, int outputNum) {
        if (AF.equals(ReLU) || AF.equals(LeakyReLU)) return () -> HE_Initialization.apply(inputNum, outputNum);
        else return () -> XAVIER_Initialization.apply(inputNum, outputNum);
    }
}
