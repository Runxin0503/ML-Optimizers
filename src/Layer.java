/** A single layer of Neurons. Contains fully connected edges to every neuron in the previous layer */
public class Layer {

    /** The number of Neurons in this layer */
    private final int nodes;

    /**
     * A 2D matrix of weights
     * <br>Rows: The neuron {@code n} in this layer
     * <br>Columns: The weight of a synapse pointing to {@code n}
     */
    private final double[][] weights;

    /** The bias of each neuron in this layer */
    private final double[] bias;

    public Layer(int nodes,int nodesBefore) {
        this.nodes = nodes;
        this.bias = new double[nodes];
        this.weights = new double[nodes][nodesBefore];
    }

    /** Applies the weights and biases of this Layer to the given input. Returns a new array. */
    public double[] calculateWeightedOutput(double[] input){
        double[] output = new double[nodes];

        for(int i=0;i<nodes;i++){
            for(int j=0;j<input.length;j++){
                output[i] += weights[i][j]*input[j];
            }
            output[i] += bias[i];
        }

        return output;
    }

    /**
     * Given the derivative array of the latest input sum,
     * calculates and shifts the given weight and bias gradients.
     * @param weightGradient Rows: The neuron n stored in that layer <br>Columns: The weight deriv of a synapse pointing to n
     * @param biasGradiant Index: The bias deriv of a node in that layer
     * @return the array of derivatives of previous layer's activation function with respect to loss function
     */
    public double[] updateGradient(double[][] weightGradient, double[] biasGradiant, double[] layerInputSumDeriv, double[] latestInput){
        double[] activationFunctionDerivative = new double[weights[0].length];
        for(int i=0;i<nodes;i++){
            for(int j=0;j<weights[0].length;j++){
                weightGradient[i][j] += latestInput[j] * layerInputSumDeriv[i];
                activationFunctionDerivative[j] += layerInputSumDeriv[i] * weights[i][j];
            }
            biasGradiant[i] += layerInputSumDeriv[i];
        }
        return activationFunctionDerivative;
    }

    /** Return the number of Neurons contained in this Layer */
    public int getNumNodes() {return nodes;}

    /**
     * Applies the {@code weightGradient} and {@code biasGradient} to the weight and bias of this Layer.
     * <br>Updates the weight and bias's gradient velocity vectors accordingly as well.
     */
    public void applyGradiant(double[][] weightGradient,double[] biasGradient,double adjustedLearningRate,double momentum){
        for(int i=0;i<nodes;i++){
            for(int j=0;j<weights[0].length;j++){
                weights[i][j] -= adjustedLearningRate * weightGradient[i][j];
            }
            bias[i] -= adjustedLearningRate * biasGradient[i];
        }
    }
}
