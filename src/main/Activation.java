package main;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

/** Activation Function enum containing regular and derivative functions of commonly-used Activation Functions */
public enum Activation {
    none(input -> {
        double[] output = new double[input.length];
        System.arraycopy(input,0,output,0,input.length);
        return output;
    },(input,gradient) -> {
        double[] output = new double[input.length];
        for(int i=0;i<input.length;i++) output[i] = input[i] * gradient[i];
        return output;
    }),
    ReLU(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = (input[i] > 0 ? input[i] : 0);
        return output;
    }, (input,gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = gradient[i] * (input[i] > 0 ? 1.0 : 0);
        return output;
    }),
    sigmoid(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = 1 / (1 + Math.pow(Math.E, -input[i]));
        return output;
    }, (input,gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++){
            double a = 1/(1+Math.pow(Math.E,-input[i]));
            output[i] = gradient[i] * a * (1-a);
        }
        return output;
    }),
    tanh(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++)
            output[i] = (Math.pow(Math.E, input[i]) - Math.pow(Math.E, -input[i])) / (Math.pow(Math.E, input[i]) + Math.pow(Math.E, -input[i]));
        return output;
    },(input,gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = gradient[i] * (1-Math.pow((Math.pow(Math.E, input[i])-Math.pow(Math.E,-input[i]))/(Math.pow(Math.E, input[i])+Math.pow(Math.E,-input[i])),2));
        return output;
    }),
    leakyReLU(input -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) output[i] = Math.max(input[i], 0.1 * input[i]);
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for(int i = 0; i < input.length; i++) output[i] = gradient[i] * (input[i] > 0 ? 1.0 : 0.1);
        return output;
    }),
    softmax(input -> {
        double[] output = new double[input.length];
        double latestInputSum = 0,max = Double.MIN_VALUE;
        for (double num : input) max = Math.max(max, num);
        for (double num : input) latestInputSum += Math.exp(num - max);
        for (int i = 0; i < input.length; i++) output[i] = Math.exp(input[i] - max) / latestInputSum;
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        double[] softmaxOutput = new double[input.length];
        double latestInputSum = 0,max = Double.MIN_VALUE;
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

    private final Function<double[],double[]> function;
    private final BiFunction<double[],double[],double[]> derivativeFunction;
    Activation(Function<double[],double[]> function,BiFunction<double[],double[],double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Returns the result of AF(x) for every x in {@code input} array*/
    public double[] calculate(double[] input) {
        for(double v : input) assert Double.isFinite(v) : "Attempted to input invalid values into Activation Function";
        double[] output = this.function.apply(input);
        for(double v : output) assert Double.isFinite(v) : "Activation Function returning invalid values " + Arrays.toString(input);
        return output;
    }

    /**
     * Effect: multiplies each element in {@code da_dC[i]} with their corresponding element {@code AF'(z[i])}
     * @return {@code dz_dC}
     */
    public double[] derivative(double[] z, double[] da_dC) {
        for(double v : da_dC) assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Activation Function";
        double[] newGradient = this.derivativeFunction.apply(z, da_dC);
        for(double v : newGradient) assert Double.isFinite(v) : "Deriv of Activation Function returning invalid values "  + Arrays.toString(z) + "  " + Arrays.toString(da_dC);
        return newGradient;
    }
}
