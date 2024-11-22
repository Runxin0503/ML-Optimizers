import java.util.function.BiConsumer;

/** Cost Function enum containing both regular and derivative version of commonly-used Loss Functions */
public enum Cost {
    diffSquared((output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = (output[i]-expectedOutput[i]) * (output[i]-expectedOutput[i]);
    },(output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = 2*(output[i]-expectedOutput[i]);
    }),
    crossEntropy((output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) {
            assert output[i] > 0 : "crossEntropy loss function must take in positive, non-zero values";
            output[i] = -(expectedOutput[i]*Math.log(output[i])+(1-expectedOutput[i])*Math.log(1-output[i]));
        }
    },(output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = -((output[i]==0 ? 0 : expectedOutput[i]/output[i]) - (1-expectedOutput[i])*(1-output[i]));
    });

    private final BiConsumer<double[],double[]> function;
    private final BiConsumer<double[],double[]> derivativeFunction;
    Cost(BiConsumer<double[],double[]> function,BiConsumer<double[],double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Transform {@code output} to the result of applying this Cost Function on the given output and expectedOutput array */
    public void calculate(double[] output,double[] expectedOutput) {
        for(double v : output) assert Double.isFinite(v) : "Attempted to input invalid values into Loss Function";
        this.function.accept(output,expectedOutput);
        for(double v : output) assert Double.isFinite(v) : "Loss Function returning invalid values";
    }

    /** Transform {@code output} to the result of applying the derivative of this Cost Function on the given output and expectedOutput array */
    public void derivative(double[] output,double[] expectedOutput) {
        for(double v : output) assert Double.isFinite(v) : "Attempted to input invalid values into Deriv of Loss Function";
        this.derivativeFunction.accept(output,expectedOutput);
        for(double v : output) assert Double.isFinite(v) : "Deriv of Loss Function returning invalid values";
    }
}
