import java.util.function.BiConsumer;

/** Cost Function enum containing both regular and derivative version of commonly-used Loss Functions */
public enum Cost {
    diffSquared((output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = Math.pow(output[i]-expectedOutput[i],2);
    },(output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = 2*(output[i]-expectedOutput[i]);
    }),
    crossEntropy((output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = -(expectedOutput[i]*Math.log(output[i])+(1-expectedOutput[i])*Math.log(1-output[i]));
    },(output,expectedOutput) ->{
        for(int i=0;i<output.length;i++) output[i] = -(expectedOutput[i]/output[i] - (1-expectedOutput[i])*(1-output[i]));
    });

    private final BiConsumer<double[],double[]> function;
    private final BiConsumer<double[],double[]> derivativeFunction;
    Cost(BiConsumer<double[],double[]> function,BiConsumer<double[],double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Transform {@code output} to the result of applying this Cost Function on the given output and expectedOutput array */
    public void calculate(double[] output,double[] expectedOutput) {this.function.accept(output,expectedOutput);}

    /** Transform {@code output} to the result of applying the derivative of this Cost Function on the given output and expectedOutput array */
    public void derivative(double[] output,double[] expectedOutput) {this.derivativeFunction.accept(output,expectedOutput);}
}
