import java.util.Arrays;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/** Activation Function enum containing regular and derivative functions of commonly-used Activation Functions */
public enum Activation {
    none(input -> {
    },(output,gradient) -> {
    }),
    ReLU(input -> {
        for (int i = 0; i < input.length; i++) input[i] = (input[i] > 0 ? input[i] : 0);
    }, (output,gradient) -> {
        for (int i = 0; i < output.length; i++) output[i] = gradient[i] * (output[i] > 0 ? 1.0 : 0);
    }),
    sigmoid(input -> {
        for (int i = 0; i < input.length; i++) input[i] = 1 / (1 + Math.pow(Math.E, -input[i]));
    }, (output,gradient) -> {
        for (int i = 0; i < output.length; i++){
            double a = 1/(1+Math.pow(Math.E,-output[i]));
            output[i] = gradient[i] * a * (1-a);
        }
    }),
    tanh(input -> {
        for (int i = 0; i < input.length; i++)
            input[i] = (Math.pow(Math.E, input[i]) - Math.pow(Math.E, -input[i])) / (Math.pow(Math.E, input[i]) + Math.pow(Math.E, -input[i]));
    },(output,gradient) -> {
        for (int i = 0; i < output.length; i++) output[i] = gradient[i] * (1-Math.pow((Math.pow(Math.E, output[i])-Math.pow(Math.E,-output[i]))/(Math.pow(Math.E, output[i])+Math.pow(Math.E,-output[i])),2));
    }),
    leakyReLU(input -> {
        for (int i = 0; i < input.length; i++) input[i] = Math.max(input[i], 0.1 * input[i]);
    }, (output, gradient) -> {
        for(int i = 0; i < output.length; i++) output[i] = gradient[i] * (output[i] > 0 ? 1.0 : 0.1);
    }),
    softmax(input -> {
        double latestInputSum = 0;
        for (double num : input) latestInputSum += Math.exp(num);
        for (int i = 0; i < input.length; i++) input[i] = Math.exp(input[i]) / latestInputSum;
    }, (output, gradient) -> {
        for (int i = 0; i < output.length; i++) {
            double val = 0;
            for (int j = 0; j < output.length; j++) {
                if (i == j) val += gradient[j] * output[i] * (1 - output[i]);
                else val += gradient[j] * (-output[i] * output[j]);
            }
            gradient[i] = val;
        }
    });

    private final Consumer<double[]> function;
    private final BiConsumer<double[],double[]> derivativeFunction;
    Activation(Consumer<double[]> function,BiConsumer<double[],double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Transform the given input array to the result of applying this Activation Function on that array*/
    public void calculate(double[] input) {
        for(double v : input) assert Double.isFinite(v) : "Attempted to input invalid values into Activation Function";
        this.function.accept(input);
        for(double v : input) assert Double.isFinite(v) : "Activation Function returning invalid values";
    }

    /** Transform the given {@code gradient} by multiplying each element with the result of applying the derivative of this Activation Function onto the {@code output} array */
    public void derivative(double[] output,double[] gradient) {
        for(double v : gradient) assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Activation Function";
        this.derivativeFunction.accept(output,gradient);
        for(double v : gradient) assert Double.isFinite(v) : "Deriv of Activation Function returning invalid values";
    }
}
