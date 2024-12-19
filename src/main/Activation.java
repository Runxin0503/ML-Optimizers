package main;

import java.util.Arrays;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
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
        double latestInputSum = 0,max = Integer.MIN_VALUE;
        for (double num : input) max = Math.max(max, num);
        for (double num : input) latestInputSum += Math.exp(num - max);
        for (int i = 0; i < input.length; i++) output[i] = Math.exp(input[i] - max) / latestInputSum;
        return output;
    }, (input, gradient) -> {
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            double val = 0;
            for (int j = 0; j < input.length; j++) {
                if (i == j) val += gradient[j] * input[i] * (1 - input[i]);
                else val += gradient[j] * (-input[i] * input[j]);
            }
            output[i] = val;
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
        for(double v : output) assert Double.isFinite(v) : "Activation Function returning invalid values from input - " + Arrays.toString(input);
        return output;
    }

    /**
     * Returns the result of multiplying each element in {@code gradient} with AF'(x) for every x in {@code input}
     */
    public double[] derivative(double[] input,double[] gradient) {
        for(double v : gradient) assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Activation Function";
        double[] newGradient = this.derivativeFunction.apply(input,gradient);
        for(double v : newGradient) assert Double.isFinite(v) : "Deriv of Activation Function returning invalid values";
        return newGradient;
    }
}
