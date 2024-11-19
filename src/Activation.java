import java.util.function.Consumer;

/** Activation Function enum containing regular and derivative functions of commonly-used Activation Functions */
public enum Activation {
    none(input -> {
    },input -> {}),
    ReLU(input -> {
        for (int i = 0; i < input.length; i++) input[i] = (input[i] > 0 ? input[i] : 0);
    }, input -> {
        for (int i = 0; i < input.length; i++) input[i] = (input[i] > 0 ? 1.0 : 0);
    }),
    sigmoid(input -> {
        for (int i = 0; i < input.length; i++) input[i] = 1 / (1 + Math.pow(Math.E, -input[i]));
    }, input -> {
        for (int i = 0; i < input.length; i++){
            double a = 1/(1+Math.pow(Math.E,-input[i]));
            input[i] = a * (1-a);
        }
    }),
    tanh(input -> {
        for (int i = 0; i < input.length; i++)
            input[i] = (Math.pow(Math.E, input[i]) - Math.pow(Math.E, -input[i])) / (Math.pow(Math.E, input[i]) + Math.pow(Math.E, -input[i]));
    },input -> {
        for (int i = 0; i < input.length; i++) input[i] = 1-Math.pow((Math.pow(Math.E,input[i])-Math.pow(Math.E,-input[i]))/(Math.pow(Math.E,input[i])+Math.pow(Math.E,-input[i])),2);
    }),
    leakyReLU(input -> {
        for (int i = 0; i < input.length; i++) input[i] = Math.max(input[i], 0.1 * input[i]);
    }, input -> {
        for(int i = 0; i < input.length; i++) input[i] = (input[i] > 0 ? 1.0 : 0.1);
    }),
    softmax(input -> {
        double latestInputSum = 0;
        for (double num : input) latestInputSum += num;
        for (int i = 0; i < input.length; i++) input[i] /= latestInputSum;
    }, input -> {
        double latestInputSum = 0;
        for (double num : input) latestInputSum += num;
        for (int i = 0; i < input.length; i++) input[i] = (input[i] * latestInputSum - input[i] * input[i]) / (latestInputSum * latestInputSum);
    });

    private final Consumer<double[]> function;
    private final Consumer<double[]> derivativeFunction;
    Activation(Consumer<double[]> function,Consumer<double[]> derivativeFunction) {
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    /** Returns the result of applying this Activation Function onto the given input array */
    public void calculate(double[] input) {this.function.accept(input);}

    /** Returns the result of applying the derivative of this Activation Function onto the given input array */
    public void derivative(double[] input) {this.derivativeFunction.accept(input);}
}
