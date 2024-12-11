package tests;

import org.junit.jupiter.api.Test;

import main.Activation;
import main.Cost;
import main.NN;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class BackPropTest {

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @Test
    void trainNOTNeuralNetwork() {
        final NN linearNN = new NN(Activation.ReLU,Activation.softmax, Cost.crossEntropy,1,4,2);
        final int iterations = 10000;

        for(int i=0;i<iterations;i++) {
            int testInput = Math.random() > 0.5 ? 1 : 0;
            double[] testOutput = new double[2];
            testOutput[testInput == 1 ? 0 : 1] = 1;

            NN.learn(linearNN, 0.5, 0.9, new double[][]{{testInput}}, new double[][]{testOutput});
        }

        evaluate(1e-2,new double[][]{{1,0},{0,1}},
                linearNN.calculateOutput(new double[]{1}),
                linearNN.calculateOutput(new double[]{0}));
    }

    /** Test Procedure: AND. When input is both 1, predict 1, otherwise predict 0 */
    @Test
    void trainANDNeuralNetwork() {
        final NN linearNN = new NN(Activation.ReLU,Activation.softmax,Cost.crossEntropy,2,4,2);
        final int iterations = 10000;

        for(int i=0;i<iterations;i++) {
            double[] testInput = new double[]{Math.round(Math.random()),Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0]==1 && testInput[1]==1 ? 1 : 0] = 1;

            NN.learn(linearNN, 0.5, 0.9, new double[][]{testInput}, new double[][]{testOutput});
        }

        evaluate(1e-2,new double[][]{{1,0},{1,0},{1,0},{0,1}},
                linearNN.calculateOutput(new double[]{0,0}),
                linearNN.calculateOutput(new double[]{0,1}),
                linearNN.calculateOutput(new double[]{1,0}),
                linearNN.calculateOutput(new double[]{1,1}));
    }

    /** Test Procedure: OR. When either input is 1, predict 1, otherwise predict 0 */
    @Test
    void trainORNeuralNetwork() {
        final NN linearNN = new NN(Activation.ReLU,Activation.softmax,Cost.crossEntropy,2,4,2);
        final int iterations = 10000;

        for(int i=0;i<iterations;i++) {
            double[] testInput = new double[]{Math.round(Math.random()),Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0]==1 || testInput[1]==1 ? 1 : 0] = 1;

            NN.learn(linearNN, 0.8, 0.9, new double[][]{testInput}, new double[][]{testOutput});
        }

        evaluate(1e-2,new double[][]{{1,0},{0,1},{0,1},{0,1}},
                linearNN.calculateOutput(new double[]{0,0}),
                linearNN.calculateOutput(new double[]{0,1}),
                linearNN.calculateOutput(new double[]{1,0}),
                linearNN.calculateOutput(new double[]{1,1}));
    }

    /** Test Procedure: XOR. When both inputs are 1,1 or 0,0 predict 0, otherwise predict 1 */
    @Test
    void trainSemiComplexNeuralNetwork() {
        final NN semiComplexNN = new NN(Activation.ReLU,Activation.sigmoid,Cost.crossEntropy,2,2,1);
        final int iterations = 10_000;

        for(int i=0;i<iterations;i++) {
            double[] testInput = new double[]{Math.round(Math.random()),Math.round(Math.random())};
            double[] testOutput = new double[1];
            testOutput[0] = testInput[0]==testInput[1] ? 0 : 1;
//            System.out.println(Arrays.toString(testInput));

            NN.learn(semiComplexNN, 1, 0.9, new double[][]{testInput}, new double[][]{testOutput});
        }
//        System.out.println();

        evaluate(1e-2,new double[][]{{0},{0},{1},{1}},
                semiComplexNN.calculateOutput(new double[]{1,1}),
                semiComplexNN.calculateOutput(new double[]{0,0}),
                semiComplexNN.calculateOutput(new double[]{0,1}),
                semiComplexNN.calculateOutput(new double[]{1,0}));
    }

    private void evaluate(double threshold,double[][] expectedOutputs,double[]... actualOutputs){
        assert expectedOutputs.length == actualOutputs.length;

        for(int i=0;i<expectedOutputs.length;i++){
            double[] actualOutput = actualOutputs[i],expectedOutput = expectedOutputs[i];
            System.out.println(Arrays.toString(actualOutput));

            for(int j=0;j<actualOutput.length;j++)
                assertEquals(expectedOutput[j],actualOutput[j],threshold);
        }
    }
}
