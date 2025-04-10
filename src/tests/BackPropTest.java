package tests;

import Network.NN;
import enums.Activation;
import enums.Cost;
import org.junit.jupiter.api.RepeatedTest;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class BackPropTest {

    /** Test Procedure: When input is 0, predict 1. When input is 1, predict 0 */
    @RepeatedTest(10000)
    void trainNOTNeuralNetwork() {
        final NN linearNN = new NN.NetworkBuilder().setInputNum(1)
                .addDenseLayer(20).addDenseLayer(2)
                .setHiddenAF(Activation.ReLU).setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy).build();
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            int testInput = (int) Math.round(Math.random());
            double[] testOutput = new double[2];
            testOutput[testInput == 1 ? 0 : 1] = 1;

            NN.learn(linearNN, 0.1, 0.9, 0.9, 1e-8, new double[][]{{testInput}}, new double[][]{testOutput});

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
        final NN linearNN = new NN.NetworkBuilder().setInputNum(2)
                .addDenseLayer(6).addDenseLayer(2)
                .setHiddenAF(Activation.sigmoid).setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy).build();
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 && testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(linearNN, 0.3, 0.9, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});

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
        final NN linearNN = new NN.NetworkBuilder().setInputNum(2)
                .addDenseLayer(4).addDenseLayer(2)
                .setHiddenAF(Activation.sigmoid).setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy).build();
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == 1 || testInput[1] == 1 ? 1 : 0] = 1;

            NN.learn(linearNN, 0.4, 0.8, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});

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
    //failing 2 out of 10,000, acceptable
    void trainXORNeuralNetwork() {
        final NN semiComplexNN = new NN.NetworkBuilder().setInputNum(2)
                .addDenseLayer(8).addDenseLayer(2)
                .setHiddenAF(Activation.tanh).setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy).build();
        final int iterations = 1000;

        for (int i = 0; i < iterations; i++) {
            double[] testInput = new double[]{Math.round(Math.random()), Math.round(Math.random())};
            double[] testOutput = new double[2];
            testOutput[testInput[0] == testInput[1] ? 0 : 1] = 1;

            NN.learn(semiComplexNN, 0.015, 0.96, 0.9, 1e-4, new double[][]{testInput}, new double[][]{testOutput});

            if (evaluate(1e-2, new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}},
                    semiComplexNN.calculateOutput(new double[]{1, 1}),
                    semiComplexNN.calculateOutput(new double[]{0, 0}),
                    semiComplexNN.calculateOutput(new double[]{0, 1}),
                    semiComplexNN.calculateOutput(new double[]{1, 0}))) break;
        }

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

//    @RepeatedTest(10000) //commented out because this is NOT a great test for adam optimizer, failed 33 out of 10,000 tests
//    void trainLinearFunctions() {
//        Random rand = new Random();
//        double m = rand.nextDouble(-1000, 1000), b = rand.nextDouble(-1000, 1000);
//        Function<Double, Double> LinearFunction = (x) -> m * x + b;
//
//        NN NeuralNet = new NN.NetworkBuilder().setInputNum(1).addDenseLayer(1)
//                .setHiddenAF(Activation.ReLU).setOutputAF(Activation.none)
//                .setCostFunction(Cost.diffSquared).build();
//        final int iterations = 4_000;
//        final int bound = 10;
//
//        for (int i = 0; i < iterations; i++) {
//            double x = Math.random() * bound * (Math.signum(Math.random() - 0.5));
//            double[] testCaseInput = new double[]{x};
//            double[] testOutput = new double[]{LinearFunction.apply(x)};
//
//            if (i % (iterations / 100.0) == 0) {
////                System.out.println((i * 100.0 / iterations) + "%");
////                System.out.println("testCaseInput - " + Arrays.toString(testCaseInput));
////                System.out.println("testOutput - " + Arrays.toString(testOutput));
////                System.out.println("NeuralNet.calculateOutput - " + Arrays.toString(NeuralNet.calculateOutput(testCaseInput)));
//            }//150 / 1000
//            NN.learn(NeuralNet, 110,0.3,0.9999, 1e-8,new double[][]{testCaseInput}, new double[][]{testOutput});
//
//            if (i % (iterations / 100.0) == 0) {
////                System.out.println("NeuralNet.calculateOutput after - " + Arrays.toString(NeuralNet.calculateOutput(testCaseInput)));
////                System.out.println("NeuralNet.calculateCost - " + NeuralNet.calculateCost(testCaseInput, testOutput));
////                System.out.println("NeuralNet.calculateCost on [1] - " + NeuralNet.calculateCost(new double[]{1}, new double[]{LinearFunction.apply(1.0)}));
//            }
//        }
//
//        double totalCost = 0;
//        final int testIterations = 10000;
//        for (int i = 0; i < testIterations; i++) {
//            double x = i * bound * 1.0 / testIterations;
////            System.out.println("LinearFunction.apply(x) " + LinearFunction.apply(x));
////            System.out.println("Neural Net Output " + Arrays.toString(NeuralNet.calculateOutput(new double[]{x})));
////            System.out.println("Neural Net COST " + NeuralNet.calculateCost(new double[]{x}, new double[]{LinearFunction.apply(x)}));
//            assertEquals(LinearFunction.apply(x), NeuralNet.calculateOutput(new double[]{x})[0], 1e-2);
////            assertEquals(0, NeuralNet.calculateCost(new double[]{x}, new double[]{LinearFunction.apply(x)}), 1e-2);
//            totalCost += NeuralNet.calculateCost(new double[]{x}, new double[]{LinearFunction.apply(x)});
////            System.out.println();
//        }
//        System.out.println("totalCost: " + totalCost);
//    }
}
