package main;

import java.util.function.BiFunction;

/** Cost Function enum containing both regular and derivative version of commonly-used Loss Functions */
public enum Cost {
    diffSquared((input,expectedInput) ->{
        double[] output = new double[input.length];
        for(int i=0;i<input.length;i++) output[i] = (input[i]-expectedInput[i]) * (input[i]-expectedInput[i]) / input.length;
        return output;
    },(input,expectedInput) ->{
        double[] output = new double[input.length];
        for(int i=0;i<input.length;i++) output[i] = 2*(input[i]-expectedInput[i]) / input.length;
        return output;
    }),
    crossEntropy((input,expectedInput) ->{
        double[] output = new double[input.length];
        for(int i=0;i<input.length;i++)
            output[i] = -(expectedInput[i]==1 ? Math.log(input[i]) : Math.log(1-input[i]));
        return output;
    },(input,expectedInput) ->{
        double[] output = new double[input.length];
        for(int i=0;i<input.length;i++) output[i] = -((input[i]==0 ? 0 : expectedInput[i]/input[i]) - (1-expectedInput[i])*(1-input[i]));
        return output;
    });

    private final BiFunction<double[],double[],double[]> function;
    private final BiFunction<double[],double[],double[]> derivativeFunction;
    Cost(BiFunction<double[],double[],double[]> function,BiFunction<double[],double[],double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Transform {@code output} to the result of applying this Cost Function on the given output and expectedOutput array */
    public double[] calculate(double[] input,double[] expectedInput) {
        for(double v : input) assert Double.isFinite(v) : "Attempted to input invalid values into Loss Function";
        double[] output = this.function.apply(input,expectedInput);
        for(double v : output) assert Double.isFinite(v) : "Loss Function returning invalid values";
        return output;
    }

    /** Transform {@code output} to the result of applying the derivative of this Cost Function on the given output and expectedOutput array */
    public double[] derivative(double[] input, double[] expectedInput) {
        for(double v : input) assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Loss Function";
        double[] output = this.derivativeFunction.apply(input, expectedInput);
        for(double v : input) assert Double.isFinite(v) : "Deriv of Loss Function returning invalid values";
        return output;
    }
}
