package Network;

import java.util.function.BiFunction;

/**
 * Enum representing commonly used loss (cost) functions and their derivatives.
 * <p>
 * Each cost function constant includes:
 * <ul>
 *     <li>The forward pass cost function, calculating the loss between output and expected values.</li>
 *     <li>The derivative of the cost function, used for backpropagation during training.</li>
 * </ul>
 * This enum supports two common cost functions:
 * <ul>
 *     <li>Mean Squared Error (MSE)</li>
 *     <li>Cross-Entropy Loss</li>
 * </ul>
 */
public enum Cost {

    /**
     * Mean Squared Error: f(x, y) = (x - y)^2 / n
     * <p>
     * Where x is the output, y is the expected output, and n is the number of elements.
     */
    diffSquared((input, expectedInput) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
            output[i] = (input[i] - expectedInput[i]) * (input[i] - expectedInput[i]) / input.length;
        return output;
    }, (input, expectedInput) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = 2 * (input[i] - expectedInput[i]) / input.length;
        return output;
    }),

    /**
     * Cross-Entropy Loss: f(x, y) = - [y * log(x) + (1 - y) * log(1 - x)]
     * <p>
     * Where x is the predicted probability, and y is the true label (typically 0 or 1, but
     * fractional values in [0, 1] are supported via the full mixed formula).
     * <p>
     * Precondition: every {@code input[i]} (predicted probability) must lie in {@code [0, 1]}
     * and every {@code expectedInput[i]} (label) must lie in {@code [0, 1]}. The forward pass
     * applies the {@code 0 * log(0) = 0} convention at the boundaries, but combinations that
     * still produce {@code log(0)} of a non-zero coefficient -- {@code x = 0} with {@code y > 0},
     * or {@code x = 1} with {@code y < 1} -- yield {@code +Infinity} and trip the internal
     * finite-value assertion. The derivative mirrors this with {@code x == 0} and {@code x == 1}
     * guards that return a per-element gradient of {@code 0} at those singular points instead
     * of the mathematically-undefined {@code 1/0}.
     */
    crossEntropy((input, expectedInput) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double y = expectedInput[i], x = input[i];
            double termY = (y == 0) ? 0 : y * Math.log(x);
            double term1mY = (y == 1) ? 0 : (1 - y) * Math.log(1 - x);
            output[i] = -(termY + term1mY);
        }
        return output;
    }, (input, expectedInput) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
            output[i] = -((input[i] == 0 ? 0 : expectedInput[i] / input[i]) - (input[i] == 1 ? 0 : (1 - expectedInput[i]) / (1 - input[i])));
        return output;
    });

    private final BiFunction<double[], double[], double[]> function;
    private final BiFunction<double[], double[], double[]> derivativeFunction;

    Cost(BiFunction<double[], double[], double[]> function, BiFunction<double[], double[], double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /**
     * Calculates the loss (cost) for a given set of output and expected output values.
     * <p>
     * This function computes the cost based on the selected cost function (e.g., MSE, Cross-Entropy).
     * @param output          The predicted output values (e.g., from the network's output layer)
     * @param expectedOutput The ground truth expected output values
     * @return The calculated cost for each element in the arrays
     * @throws AssertionError if input or output contains non-finite values
     */
    double[] calculate(double[] output, double[] expectedOutput) {
        for (double v : output)
            if (!Double.isFinite(v))
                throw new IllegalArgumentException("Attempted to input invalid values into Loss Function");
        double[] costs = this.function.apply(output, expectedOutput);
        for (double v : costs)
            if (!Double.isFinite(v))
                throw new IllegalStateException("Loss Function returning invalid values");
        return costs;
    }

    /**
     * Computes the gradient of the cost function with respect to the output.
     * <p>
     * This is the derivative of the loss function, used during backpropagation to update the network's weights.
     * @param output          The predicted output values (e.g., from the network's output layer)
     * @param expectedOutput The ground truth expected output values
     * @return The gradient of the loss with respect to the output
     * @throws AssertionError if input or output contains non-finite values
     */
    double[] derivative(double[] output, double[] expectedOutput) {
        for (double v : output)
            if (!Double.isFinite(v))
                throw new IllegalArgumentException("Attempted to input invalid values into Deriv of Loss Function");
        double[] gradient = this.derivativeFunction.apply(output, expectedOutput);
        for (double v : gradient)
            if (!Double.isFinite(v))
                throw new IllegalStateException("Deriv of Loss Function returning invalid values");
        return gradient;
    }
}
