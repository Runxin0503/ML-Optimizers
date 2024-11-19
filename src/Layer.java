import java.util.Random;

public class Layer {

    public int nodes;
    public int nodesBefore;
    public double[][] weights;
    private double[][] velocityW;
    private double[] bias;
    private double[] velocityB;
    private double[] latestInputs;
    private double[] latestSx;
    private double[] latestAx;
    private double latestSD;
    private double latestInputSum;
    private double[] currentNodeDeriv;
    
    String function;
    String cost;

    public Layer(int nodes,int nodesBefore,String function,String cost){
        Random rand = new Random();
        this.nodes=nodes;
        this.nodesBefore = nodesBefore;
        this.function = function;
        this.cost = cost;

        this.velocityW = new double[nodes][nodesBefore];
        this.weights = new double[nodes][nodesBefore];
        for(int i=0;i<nodesBefore;i++){
            for(int j=0;j<nodes;j++){
                weights[j][i]=rand.nextGaussian();
            }
        }

        this.velocityB=new double[nodes];
        this.bias = new double[nodes];
        for(int i=0;i<nodes;i++){
            bias[i]=rand.nextGaussian();
        }

        this.latestInputs = new double[nodesBefore];
        this.latestSx = new double[nodes];
        this.latestAx = new double[nodes];
        this.latestSD = 0.0;
        this.currentNodeDeriv = new double[nodes];
    }

    /*
     * runs the weighted inputs through this layer of nodes
     * returns a list of outputs from each node
     */
    public double[] calculateWeightedOutput(double[] input){
        if(input.length!=nodesBefore){
            System.out.println("ERROR");
            return null;
        }

        latestInputs = batchNormalization(input);

        double[] output = new double[nodes];
        for(int i=0;i<nodes;i++){
            double sum=bias[i];
            for(int j=0;j<nodesBefore;j++){
                sum+=weights[i][j]*latestInputs[j];
            }
            double temp = activationFunction(sum);
            latestAx[i]=temp;
            output[i]=temp;
            latestSx[i]=sum;
            // System.out.println(temp);
        }
        return output;
    }

    /*
     * makes sure the weighted input of each node is not too big to overflow the node
     * too big numbers also creates near-zero values during backpropogation, which needs to be prevented
     */
    private double[] batchNormalization(double[] input){
        double sum=0;
        for(int i=0;i<input.length;i++){
            sum+=input[i];
        }
        latestInputSum=sum;
        double mean = sum/input.length;
        sum=0;
        for(int i=0;i<input.length;i++){
            double temp = input[i]-mean;
            sum+=temp * temp;
        }
        double SD = Math.sqrt(sum/input.length);
        double[] output = new double[input.length];
        for(int i=0;i<input.length;i++){
            output[i]=(input[i]-mean)/SD;
        }
        return output;
    }

    /*
     * a function that is applied as the last node operation
     * enables the neural network to create multiple regressions model
     */
    private double activationFunction(double num){
        if(function.toLowerCase().equals("none")){
            return num;
        }else if(function.toLowerCase().equals("relu")){
            return (num > 0) ? num : 0;
        }else if(function.toLowerCase().equals("sigmoid")){
            return 1/(1+Math.pow(Math.E,-num));
        }else if(function.toLowerCase().equals("tanh")){
            return (Math.pow(Math.E,num)-Math.pow(Math.E,-num))/(Math.pow(Math.E,num)+Math.pow(Math.E,-num));
        }else if(function.toLowerCase().equals("leaky relu")){
            return Math.max(num,0.1*num);
        }else if(function.toLowerCase().equals("softmax")){
            return num/latestInputSum;
        }
        System.out.println("ERROR");
        return num;
    }

    /*
     * Deriv of Activation Function
     * with respect to Z(x) = Weight * x + bias
     */
    private double activationFunctionDerivative(double num){
        if(function.toLowerCase().equals("none")){
            return 1.0;
        }else if(function.toLowerCase().equals("relu")){
            return (num > 0 ? 1.0 : 0);
        }else if(function.toLowerCase().equals("sigmoid")){
            double a = activationFunction(num);
            return a * (1-a);
        }else if(function.toLowerCase().equals("tanh")){
            return 1-Math.pow((Math.pow(Math.E,num)-Math.pow(Math.E,-num))/(Math.pow(Math.E,num)+Math.pow(Math.E,-num)),2);
        }else if(function.toLowerCase().equals("leaky relu")){
            return (num > 0 ? 1 : 0.1);
        }else if(function.toLowerCase().equals("softmax")){
            return (num * latestInputSum - num * num) / (latestInputSum * latestInputSum);
        }
        System.out.println("ERROR");
        return 1.0;
    }

    /*
     * returns how "wrong" the output is according to a set of equations
     */
    public double calculateCost(double output,double expectedOutputs){
        if(cost.toLowerCase().equals("differencesquared")){
            return Math.pow(output-expectedOutputs,2);
        }else if(cost.toLowerCase().equals("crossentropy")){
            double v = expectedOutputs==1 ? -Math.log(output) : -Math.log(1-output);
            return Double.isNaN(v) ? 0 : v;
        }
        System.out.println("ERROR");
        return 0;
    }

    /*
     * Deriv of Cost Function
     * with respect to Activation Function A(x)
     */
    private double calculateCostDerivative(double output,double expectedOutputs){
        if(cost.toLowerCase().equals("differencesquared")){
            return 2*(output-expectedOutputs);
        }else if(cost.toLowerCase().equals("crossentropy")){
            return output == 0.0 || output == 1.0 ? 0 : (-output + expectedOutputs) / (output * (output - 1));
        }
        System.out.println("ERROR");
        return 0;
    }

    /*
     * for calculating and memorizing its own list of current node deriv
     * for output layer only
     */
    public double[] calculateCurrentNodeDeriv(double[] expectedOutputs){
        for(int i=0;i<nodes;i++){
            // System.out.println(calculateCostDerivative(latestAx[i],expectedOutputs[i]) + " | " + activationFunctionDerivative(latestSx[i]) + " | " + latestSx[i]);
            currentNodeDeriv[i]=calculateCostDerivative(latestAx[i],expectedOutputs[i]) * activationFunctionDerivative(latestSx[i]);
            // System.out.println(currentNodeDeriv[i]);
        }
        return currentNodeDeriv;
    }

    /*
     * to use the layer after itself to calculate its own current node deriv
     * for hidden layers only
     */
    public double[] calculateCurrentNodeDeriv(Layer afterLayer,double[] nextNodeDeriv){
        for(int i=0;i<nodes;i++){
            double nodeDeriv=0;
            for(int j=0;j<afterLayer.nodes;j++){
                // System.out.println(afterLayer.weights[j][i]+"|"+nextNodeDeriv[j]);
                nodeDeriv+=afterLayer.weights[j][i]*nextNodeDeriv[j]*afterLayer.latestSD;
            }
            currentNodeDeriv[i]=nodeDeriv*activationFunctionDerivative(latestSx[i]);
        }
        return currentNodeDeriv;
    }

    /*
     * given two gradient list pointers, update each gradient accordingly with its nodeDeriv
     */
    public void updateAllGradiants(double[][] weightGradiant,double[] biasGradiant){
        for(int i=0;i<nodes;i++){
            for(int j=0;j<nodesBefore;j++){
                // System.out.println(currentNodeDeriv[i]+" | "+latestInputs[j]+" - - - - - "+nodesBefore);
                weightGradiant[i][j]+=currentNodeDeriv[i]*latestInputs[j];
            }
            biasGradiant[i]+=currentNodeDeriv[i];
        }
    }
    
    /*
     * nudges all weights on this layer with respect to the weight gradient
     * and other factors like momentum, learn rate, and batch size
     */
    public void applyWeightGradiant(double[][] weightGradiant,double learnRate,double momentum,double batchSize){
        for(int i=0;i<nodes;i++){
            for(int j=0;j<nodesBefore;j++){
                // System.out.println(weightGradiant[i][j]);
                velocityW[i][j] = momentum * velocityW[i][j] - weightGradiant[i][j] * learnRate / batchSize;
                weights[i][j]+=velocityW[i][j];
            }
        }
    }

    /*
     * nudges all biases on this layer with respect to the bias gradient
     * and other factors like momentum, learn rate, and batch size
     */
    public void applyBiasGradiant(double[] biasGradiant,double learnRate,double momentum,double batchSize){
        for(int i=0;i<nodes;i++){
            velocityB[i] = momentum * velocityB[i] - biasGradiant[i] * learnRate / batchSize;
            bias[i]+=velocityB[i];
        }
    }
}
