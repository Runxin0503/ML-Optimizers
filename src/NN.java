public class NN {
    int inputNum;
    int outputNum;
    int[] numHidden;
    int[] layers;
    Layer[] hiddenLayers;
    double[][] biasGradiant;
    double[][][] weightGradiant;

    public NN(int...layers){
        this.layers = layers;

        inputNum=layers[0];
        outputNum=layers[layers.length-1];
        numHidden = new int[layers.length-1];
        hiddenLayers = new Layer[numHidden.length];

        for(int i=1;i<layers.length;i++){
            numHidden[i-1]=layers[i];
            hiddenLayers[i-1]= i!=layers.length-1 ? new Layer(layers[i],layers[i-1],"ReLU","crossEntropy") : new Layer(layers[i],layers[i-1],"softMax","crossEntropy");
        }

        clearGradiants();
    }

    /*
     * returns the output of the neural network
     */
    public double[] calculateOutput(double[] input){
        if (input.length!=inputNum){
            System.out.println("ERROR");
            return null;
        }
        double[] pointerHelper = hiddenLayers[0].calculateWeightedOutput(input);
        for(int i=1;i<hiddenLayers.length;i++){
            pointerHelper = hiddenLayers[i].calculateWeightedOutput(pointerHelper);
        }
        return pointerHelper;
    }

    /*
     * returns the total cost one run of the neural network
     */
    public double calculateCosts(double[] input,double[] expectedOutputs){
        double[] output = calculateOutput(input);
        double sum=0;

        for(int i=0;i<output.length;i++){
            sum+=hiddenLayers[hiddenLayers.length-1].calculateCost(output[i], expectedOutputs[i]);
        }

        return sum;
    }

//---------------------------------------------------------------------------------------------------------------------------------------------

    /*
     * sets all values of the gradient variables to 0
     * Either initializes the gradient variable or clears to save memory
     */
    private void clearGradiants(){
        this.biasGradiant = new double[layers.length-1][];
        for(int i=1;i<layers.length;i++){
            biasGradiant[i-1]=new double[layers[i]];
        }
        this.weightGradiant = new double[layers.length-1][][];
        for(int i=1;i<layers.length;i++){
            weightGradiant[i-1]=new double[layers[i]][layers[i-1]];
        }
    }

    /*
     * Tells the layer to nudge their respective weights according to the gradiant vector
     * Momentum is taken into account, as is the learn rate and batch size
     */
    private void ApplyAllGradiants(double learnRate,double momentum,int batchSize){
        for(int i=0;i<hiddenLayers.length;i++){
            hiddenLayers[i].applyBiasGradiant(biasGradiant[i],learnRate,momentum,batchSize);
            hiddenLayers[i].applyWeightGradiant(weightGradiant[i], learnRate,momentum,batchSize);
        }
    }

    /*
     * Uses calculus and derivatives to find the derivatives of the cost function with respect to each weight.
     * Uses that information to generate the gradient vector, which is later applied to nudge all weights in the direction of lowest cost
     */
    private void backPropogation(double[] inputs,double[] expectedOutputs){
        calculateOutput(inputs);

        Layer outputLayer = hiddenLayers[hiddenLayers.length-1];
        double[] currentNodeDeriv = outputLayer.calculateCurrentNodeDeriv(expectedOutputs);
        outputLayer.updateAllGradiants(weightGradiant[hiddenLayers.length-1],biasGradiant[hiddenLayers.length-1]);

        for(int j=hiddenLayers.length-2;j>=0;j--){
            Layer hiddenLayer = hiddenLayers[j];
            currentNodeDeriv = hiddenLayer.calculateCurrentNodeDeriv(hiddenLayers[j+1],currentNodeDeriv);
            hiddenLayer.updateAllGradiants(weightGradiant[j],biasGradiant[j]);
        }
    }

    /*
     * runs the back-propogation function and adjusts all weights and biases based on the resulting gradient vector
     * only works with known outputs
     */
    public void learn(double learnRate,double momentum,double[][] inputs,double[][] expectedOutputs){
        if(inputs.length!=expectedOutputs.length){
            System.out.println("ERROR");
            return;
        }

        for(int i=0;i<inputs.length;i++){
            backPropogation(inputs[i],expectedOutputs[i]);
        }

        ApplyAllGradiants(learnRate,momentum,inputs.length);

        clearGradiants();
    }
}
