package tests;

import main.Activation;
import main.Cost;
import main.NN;
import org.junit.jupiter.api.RepeatedTest;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BackPropTest {

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @RepeatedTest(10000)
    void trainNOTNeuralNetwork() {
        final NN linearNN = new NN(Activation.ReLU, Activation.softmax, Cost.crossEntropy, 1,20,2);
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            int testInput = (int) Math.round(Math.random());
            double[] testOutput = new double[2];
            testOutput[testInput == 1 ? 0 : 1] = 1;

            NN.learn(linearNN, 0.2, 0.9, new double[][]{{testInput}}, new double[][]{testOutput});

            if (evaluate(1e-2, new double[][]{{1, 0}, {0, 1}},
                    linearNN.calculateOutput(new double[]{1}),
                    linearNN.calculateOutput(new double[]{0}))) break;
        }

        assertTrue(evaluate(1e-2, new double[][]{{1, 0}, {0, 1}},
                linearNN.calculateOutput(new double[]{1}),
                linearNN.calculateOutput(new double[]{0})));
    }

    /** Test Procedure: AND. When input is both 1, predict 1, otherwise predict 0 */
    @RepeatedTest(10000)
    void trainANDNeuralNetwork() {
        final NN linearNN = new NN(Activation.sigmoid, Activation.softmax, Cost.crossEntropy, 2, 4, 2);
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(linearNN, 0.75, 0.9, new double[][]{testInput}, new double[][]{testOutput});

            if (evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}},
                    linearNN.calculateOutput(new double[]{0, 0}),
                    linearNN.calculateOutput(new double[]{0, 1}),
                    linearNN.calculateOutput(new double[]{1, 0}),
                    linearNN.calculateOutput(new double[]{1, 1}))) break;
        }

        assertTrue(evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {1, 0}, {0, 1}},
                linearNN.calculateOutput(new double[]{0, 0}),
                linearNN.calculateOutput(new double[]{0, 1}),
                linearNN.calculateOutput(new double[]{1, 0}),
                linearNN.calculateOutput(new double[]{1, 1})));
    }

    /** Test Procedure: OR. When either input is 1, predict 1, otherwise predict 0 */
    @RepeatedTest(10000)
    void trainORNeuralNetwork() {
        final NN linearNN = new NN(Activation.sigmoid, Activation.softmax, Cost.crossEntropy, 2, 4, 2);
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 || testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(linearNN, 0.8, 0.9, new double[][]{testInput}, new double[][]{testOutput});

            if (evaluate(1e-2, new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}},
                    linearNN.calculateOutput(new double[]{0, 0}),
                    linearNN.calculateOutput(new double[]{0, 1}),
                    linearNN.calculateOutput(new double[]{1, 0}),
                    linearNN.calculateOutput(new double[]{1, 1}))) break;
        }

        assertTrue(evaluate(1e-2, new double[][]{{1, 0}, {0, 1}, {0, 1}, {0, 1}},
                linearNN.calculateOutput(new double[]{0, 0}),
                linearNN.calculateOutput(new double[]{0, 1}),
                linearNN.calculateOutput(new double[]{1, 0}),
                linearNN.calculateOutput(new double[]{1, 1})));
    }

    /** Test Procedure: XOR. When both inputs are 1,1 or 0,0 predict 0, otherwise predict 1 */
    @RepeatedTest(10000)
    void trainXORNeuralNetwork() { //todo 1 out of 10,000 failed
        final NN semiComplexNN = new NN(Activation.tanh, Activation.softmax, Cost.crossEntropy, 2, 8, 2);
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == testInput[1] ? 0 : 1] = 1;
//            System.out.println(Arrays.toString(testInput));

            NN.learn(semiComplexNN, 0.5, 0.9, new double[][]{testInput}, new double[][]{testOutput});

            if (evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}},
                    semiComplexNN.calculateOutput(new double[]{1, 1}),
                    semiComplexNN.calculateOutput(new double[]{0, 0}),
                    semiComplexNN.calculateOutput(new double[]{0, 1}),
                    semiComplexNN.calculateOutput(new double[]{1, 0}))) break;
        }
//        System.out.println();

        assertTrue(evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}},
                semiComplexNN.calculateOutput(new double[]{1, 1}),
                semiComplexNN.calculateOutput(new double[]{0, 0}),
                semiComplexNN.calculateOutput(new double[]{0, 1}),
                semiComplexNN.calculateOutput(new double[]{1, 0})));
    }

    private boolean evaluate(double threshold, double[][] expectedOutputs, double[]... actualOutputs) {
        assert expectedOutputs.length == actualOutputs.length;

        for (int i = 0; i < expectedOutputs.length; i++) {
            double[] actualOutput = actualOutputs[i], expectedOutput = expectedOutputs[i];

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
        final int iterations = 1_000;
        final int bound = 10;

        for (int i = 0; i < iterations; i++) {
            double x = Math.random() * bound * (Math.signum(Math.random() - 0.5));
            double[] testCaseInput = new double[]{x};
            double[] testOutput = new double[]{LinearFunction.apply(x)};

            if (i % (iterations / 100.0) == 0) {
                System.out.println((i * 100.0 / iterations) + "%");
                System.out.println("testCaseInput - " + Arrays.toString(testCaseInput));
                System.out.println("testOutput - " + Arrays.toString(testOutput));
                System.out.println("NeuralNet.calculateOutput - " + Arrays.toString(NeuralNet.calculateOutput(testCaseInput)));
            }

            NN.learn(NeuralNet, 0.01, 0.9, new double[][]{testCaseInput}, new double[][]{testOutput});

            if (i % (iterations / 100.0) == 0) {
                System.out.println("NeuralNet.calculateOutput after - " + Arrays.toString(NeuralNet.calculateOutput(testCaseInput)));
                System.out.println("NeuralNet.calculateCost - " + NeuralNet.calculateCosts(testCaseInput, testOutput));
                System.out.println("NeuralNet.calculateCost on [1] - " + NeuralNet.calculateCosts(new double[]{1}, new double[]{LinearFunction.apply(1.0)}));
            }
        }

        double totalCost = 0;
        final int testIterations = 10000;
        for (int i = 0; i < testIterations; i++) {
            double x = i * bound * 1.0 / testIterations;
            System.out.println("LinearFunction.apply(x) " + LinearFunction.apply(x));
            System.out.println("Neural Net Output " + Arrays.toString(NeuralNet.calculateOutput(new double[]{x})));
            System.out.println("Neural Net COST " + NeuralNet.calculateCosts(new double[]{x}, new double[]{LinearFunction.apply(x)}));
            assertEquals(LinearFunction.apply(x), NeuralNet.calculateOutput(new double[]{x})[0], 1e-2);
            assertEquals(0, NeuralNet.calculateCosts(new double[]{x}, new double[]{LinearFunction.apply(x)}), 1e-2);
            totalCost += NeuralNet.calculateCosts(new double[]{x}, new double[]{LinearFunction.apply(x)});
            System.out.println();
        }
        System.out.println("totalCost: " + totalCost);
    }
}
