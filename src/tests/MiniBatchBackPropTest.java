package tests;

import main.Activation;
import main.Cost;
import main.NN;
import org.junit.jupiter.api.RepeatedTest;
import java.util.Random;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MiniBatchBackPropTest {

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @RepeatedTest(10000)
    void trainNOTNeuralNetwork() {
        final NN linearNN = new NN(Activation.ReLU, Activation.softmax, Cost.crossEntropy, 1, 20, 2);
        final int iterations = 3000;

        double[][] testCaseInputs = new double[][]{{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}};
        double[][] testCaseOutputs = new double[][]{{0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}};

        for (int i = 0; i < iterations; i++) {
            NN.learn(linearNN, 1.5, 0.9,1e-4, testCaseInputs, testCaseOutputs);
            if (evaluate(testCaseInputs, testCaseOutputs, linearNN, 1e-2)) break;
        }

        assertTrue(evaluate(testCaseInputs, testCaseOutputs, linearNN, 1e-2));
    }

    /** Test Procedure: AND. When input is both 1, predict 1, otherwise predict 0 */
    @RepeatedTest(10000)
    void trainANDNeuralNetwork() {
        final NN linearNN = new NN(Activation.sigmoid, Activation.softmax, Cost.crossEntropy, 2, 4, 2);
        final int iterations = 1000;

        double[][] testCaseInputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] testCaseOutputs = new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}};

        for (int i = 0; i < iterations; i++) {
            NN.learn(linearNN, 1, 0.9, 1e-4,testCaseInputs, testCaseOutputs);
            if (evaluate(testCaseInputs, testCaseOutputs, linearNN, 1e-2)) break;
        }

        assertTrue(evaluate(testCaseInputs, testCaseOutputs, linearNN, 1e-2));

    }

    /** Test Procedure: OR. When either input is 1, predict 1, otherwise predict 0 */
    @RepeatedTest(10000)
    void trainORNeuralNetwork() {
        final NN linearNN = new NN(Activation.sigmoid, Activation.softmax, Cost.crossEntropy, 2, 4, 2);
        final int iterations = 1000;

        double[][] testCaseInputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] testCaseOutputs = new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}};

        for (int i = 0; i < iterations; i++) {
            NN.learn(linearNN, 1, 0.9, 1e-4,testCaseInputs, testCaseOutputs);
            if (evaluate(testCaseInputs, testCaseOutputs, linearNN, 1e-2)) break;
        }

        assertTrue(evaluate(testCaseInputs, testCaseOutputs, linearNN, 1e-2));
    }

    /** Test Procedure: XOR. When both inputs are 1,1 or 0,0 predict 0, otherwise predict 1 */
    @RepeatedTest(10000)
    void trainXORNeuralNetwork() {
        final NN notLinearNN = new NN(Activation.tanh, Activation.softmax, Cost.crossEntropy, 2,10, 2);
        final int iterations = 1000;

        double[][] testCaseInputs = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] testCaseOutputs = new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}};

        for (int i = 0; i < iterations; i++) {//1.25,0.95 vs 1,0.9 (10 hidden neurons)
            NN.learn(notLinearNN, 1, 0.9, 1e-4,testCaseInputs, testCaseOutputs);
            if (evaluate(testCaseInputs, testCaseOutputs, notLinearNN, 1e-2)) break;
        }

        assertTrue(evaluate(testCaseInputs, testCaseOutputs, notLinearNN, 1e-2));
    }

    private boolean evaluate(double[][] inputs, double[][] expectedOutputs, NN NeuralNet, double threshold) {
        assert inputs.length == expectedOutputs.length;

        for (int i = 0; i < inputs.length; i++) {
            double[] actualOutput = NeuralNet.calculateOutput(inputs[i]), expectedOutput = expectedOutputs[i];

            for (int j = 0; j < actualOutput.length; j++)
                if (Math.abs(expectedOutput[j] - actualOutput[j]) > threshold) {
                    System.err.println("expected: " + expectedOutput[j] + " but was: " + actualOutput[j]);
                    return false;
                }
        }
        return true;
    }

@RepeatedTest(10000)
    void trainLinearFunctions() {
        Random rand = new Random();
        double m = rand.nextDouble(-1000, 1000), b = rand.nextDouble(-1000, 1000);
        Function<Double, Double> LinearFunction = (x) -> m * x + b;

        NN NeuralNet = new NN(Activation.ReLU, Activation.none, Cost.diffSquared, 1, 1);
        final int iterations = 5_000;
        final int batchSize = 10;
        final int bound = 10;

        for (int i = 0; i < iterations; i += batchSize) {
            double[][] testCaseInputs = new double[batchSize][1];
            double[][] testCaseOutputs = new double[batchSize][1];
            for (int j = 0; j < batchSize; j++) {
                double x = Math.random() * bound * (Math.signum(Math.random() - 0.5));
                testCaseInputs[j] = new double[]{x};
                testCaseOutputs[j] = new double[]{LinearFunction.apply(x)};
            }

//            if (i % (iterations / 100.0) == 0) {
//                System.out.println((i * 100.0 / iterations) + "%");
//                System.out.print("testCaseInput - ");
//                Arrays.asList(testCaseInputs).forEach(e -> System.out.print(Arrays.toString(e) + ','));
//                System.out.print("\ntestCaseOutputs - ");
//                Arrays.asList(testCaseOutputs).forEach(e -> System.out.print(Arrays.toString(e) + ','));
//                System.out.print("\nNeuralNet.calculateOutput - ");
//                Arrays.asList(testCaseInputs).forEach(e -> System.out.print(Arrays.toString(NeuralNet.calculateOutput(e)) + ','));
//                System.out.println();
//            }

            NN.learn(NeuralNet, 100, 0.9999, 1e-4,testCaseInputs, testCaseOutputs);

//            if (i % (iterations / 100.0) == 0) {
//                System.out.print("NeuralNet.calculateOutput after - ");
//                Arrays.asList(testCaseInputs).forEach(e -> System.out.print(Arrays.toString(NeuralNet.calculateOutput(e)) + ','));
//                System.out.print("\nNeuralNet.calculateCost - ");
//                for (int j = 0; j < testCaseInputs.length; j++)
//                    System.out.print(NeuralNet.calculateCosts(testCaseInputs[j], testCaseOutputs[j]) + ',');
//                System.out.println("\nNeuralNet.calculateCost on [1] - " + NeuralNet.calculateCosts(new double[]{1}, new double[]{LinearFunction.apply(1.0)}));
//            }
        }

        double totalCost = 0;
        final int testIterations = 10000;
        for (int i = 0; i < testIterations; i++) {
            double x = i * bound * 1.0 / testIterations;
//            System.out.println("LinearFunction.apply(x) " + LinearFunction.apply(x));
//            System.out.println("Neural Net Output " + Arrays.toString(NeuralNet.calculateOutput(new double[]{x})));
//            System.out.println("Neural Net COST " + NeuralNet.calculateCosts(new double[]{x}, new double[]{LinearFunction.apply(x)}));
            assertEquals(LinearFunction.apply(x), NeuralNet.calculateOutput(new double[]{x})[0], 1e-2);
            assertEquals(0, NeuralNet.calculateCosts(new double[]{x}, new double[]{LinearFunction.apply(x)}), 1e-2);
            totalCost += NeuralNet.calculateCosts(new double[]{x}, new double[]{LinearFunction.apply(x)});
//            System.out.println();
        }
        System.out.println("totalCost: " + totalCost);
    }
}
