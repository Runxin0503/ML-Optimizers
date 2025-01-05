package tests;

import main.Activation;
import main.Cost;
import main.NN;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;

import javax.management.relation.InvalidRoleInfoException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class IrisDatasetTest {

    private static final int Iris_Size = 150;
    private static final HashMap<double[],Integer> featuresToCategories = new HashMap<>(Iris_Size);
    private static final List<double[]> features;
    private static final ArrayList<String> names = new ArrayList<>();

    static {
        try (BufferedReader br = new BufferedReader(new FileReader("lib/iris.data"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double[] features = new double[4];
                for (int i = 0; i < 4; i++) {
                    features[i] = Double.parseDouble(parts[i]);
                }
                String label = parts[4];
                if (!names.contains(label)) names.add(label);
                featuresToCategories.put(features,names.indexOf(label));
            }
            assert featuresToCategories.size() == Iris_Size;
            features = featuresToCategories.keySet().stream().toList();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @RepeatedTest(1000)
    void testDataset() {
        NN NeuralNet = new NN(Activation.sigmoid, Activation.softmax,Cost.crossEntropy, 4,10,10, names.size());

        final int iterations = 200;
        final int batchSize = 15;
        final int report_interval = 50; //3 reports per iteration
        for(int loopedIterations = 0;loopedIterations<iterations;loopedIterations++) {
            for (int trainingIndex = 0; trainingIndex < Iris_Size - batchSize; trainingIndex += batchSize) {
                double[][] trainBatchInputs = new double[batchSize][4];
                double[][] trainBatchOutputs = new double[batchSize][names.size()];
                for (int i = 0; i < batchSize; i++) {
                    trainBatchInputs[i] = features.get(trainingIndex + i);
                    trainBatchOutputs[i][featuresToCategories.get(trainBatchInputs[i])] = 1;
                }//momentum 0.9, learningRate: 0.01 -> 5%, 0.05 -> 3%
                //momentum 0.9, learningRate: 0.01 -> ?%, 0.05 -> ?% more hidden neurons
                //momentum 0.95, learningRate: 0.01 -> ?%, 0.05 -> 1% (relu)
                //momentum 0.95, learningRate: 0.01 -> bad, 0.05 -> ?%, 0.1 -> ?% (sigmoid)
                NN.learn(NeuralNet, 0.1, 0.9,1e-4, trainBatchInputs, trainBatchOutputs);

                if ((trainingIndex + batchSize) % report_interval == 0) {
//                    System.out.print("Iteration " + ((int) (((double) trainingIndex) / batchSize) + 1));
//                    System.out.println(", " + (int) ((trainingIndex + 1.0) / Iris_Size * 10000) / 100.0 + "% finished");
//                    reportPerformanceOnTrain(NeuralNet, trainingIndex);
//                    reportPerformanceOnTest(NeuralNet, trainingIndex);
//                    System.out.println("Predicted Output for " + names.get(featuresToCategories.get(features.getFirst())) + ": " + names.get(getOutput(NeuralNet.calculateOutput(features.getFirst()))));
//                    System.out.println(Arrays.toString(NeuralNet.calculateOutput(features.getFirst())));
//                    System.out.println("--------------------");
                }
            }
        }

//        for(double[] feature : features)
//            System.out.println("Predicted Output for " + names.get(featuresToCategories.get(feature)) + ":\t" + names.get(getOutput(NeuralNet.calculateOutput(feature))));

        reportPerformanceOnTest(NeuralNet,0);
    }

    /**
     * Reports the performance of {@code NeuralNet} on the first {@code n} inputs of the MNIST dataset
     * <br>In other words, evaluate the model on examples it has seen
     */
    private static void reportPerformanceOnTrain(NN NeuralNet, int n) {
        double cost = 0;
        int accuracy = 0;
        for (int i = 0; i < n; i++) {
            double[] feature = features.get(i);
            int category = featuresToCategories.get(feature);
            double[] expectedOutput = new double[10];
            expectedOutput[category] = 1;
            cost += NeuralNet.calculateCosts(feature, expectedOutput);
            if (evaluateOutput(NeuralNet.calculateOutput(feature), category)) accuracy++;
        }
        System.out.println("Train Accuracy: " + accuracy * 10000 / (n * 100.0) + "%\t\tAvg Cost: " + (int) (cost * 100) / (n * 100.0));
    }

    /**
     * Reports the performance of {@code NeuralNet} on everything BUT the first {@code n} inputs of the MNIST dataset
     * <br>In other words, evaluate the model on examples it hasn't seen yet
     */
    private static void reportPerformanceOnTest(NN NeuralNet, int n) {
        double cost = 0;
        int accuracy = 0;
        for (int i = n; i < Iris_Size; i++) {
            double[] feature = features.get(i);
            int category = featuresToCategories.get(feature);
            double[] expectedOutput = new double[10];
            expectedOutput[category] = 1;
            cost += NeuralNet.calculateCosts(feature, expectedOutput);
            if (evaluateOutput(NeuralNet.calculateOutput(feature), category)) accuracy++;
        }
        System.out.println("Test Accuracy: " + accuracy * 10000 / (Iris_Size - n) * 0.01 + "%\t\tAvg Cost: " + (int) (cost * 100) / (Iris_Size - n) * 0.01);
        assertTrue((double) accuracy / (Iris_Size - n) > 0.97);
    }

    private static boolean evaluateOutput(double[] output, int answer) {
        return getOutput(output) == answer;
    }

    private static int getOutput(double[] output) {
        int guess = 0;
        for (int j = 0; j < output.length; j++) {
            if (output[j] > output[guess]) guess = j;
        }
        return guess;
    }
}
