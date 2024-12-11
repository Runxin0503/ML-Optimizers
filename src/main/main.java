package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class main {

    public static void main(String[] args) {
        double[][] image = new double[70000][784];
        int[] answer = new int[70000];
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader("lib/MNIST DATA.csv"))) {
            String line;
            int count = 0;
            while ((line = bufferedReader.readLine()) != null) {
                String[] parts = line.split(",");
                answer[count] = Integer.parseInt(parts[0]);
                for (int i = 1; i < parts.length; i++) {
                    image[count][i - 1] = Integer.parseInt(parts[i]) / 255.0;
                }
                count++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("LETS START LEARNING!\n--------------------");

        NN NeuralNet = new NN(Activation.sigmoid, Activation.softmax, Cost.crossEntropy, 784, 100, 10);

        final int batchSize = 1;
        for (int trainingIndex = 0; trainingIndex < 63000; trainingIndex += batchSize) {
            double[][] trainBatchInputs = new double[batchSize][784];
            double[][] trainBatchOutputs = new double[batchSize][10];
            for (int i = 0; i < batchSize; i++) {
                trainBatchInputs[i] = image[trainingIndex + i];
                trainBatchOutputs[i][answer[trainingIndex + i]] = 1.0;
            }

            NN.learn(NeuralNet, 1, 0.9, trainBatchInputs, trainBatchOutputs);

            if (trainingIndex % 7000 == 0) {
                System.out.println("Iterations " + (trainingIndex * batchSize + 1));
                reportPerformanceOnTest(NeuralNet, image, answer);
                reportPerformanceOnTrain(NeuralNet, image, answer, trainingIndex);
                System.out.println("Predicted Output for " + answer[0] + ": \n" + Arrays.toString(NeuralNet.calculateOutput(trainBatchInputs[0])));
                System.out.println("--------------------");
            }
        }
    }

    private static void reportPerformanceOnTrain(NN NeuralNet, double[][] image, int[] answer, int currentTrainingIndex) {
        double batchCost = 0;
        double batchAccuracy = 0;
        for (int i = 0; i < currentTrainingIndex; i++) {
            double[] expectedOutput = new double[10];
            expectedOutput[answer[i]] = 1;
            batchCost += NeuralNet.calculateCosts(image[i], expectedOutput);
            if (evaluateOutput(NeuralNet.calculateOutput(image[i]), answer[i])) batchAccuracy++;
        }
        System.out.println("Train Avg java.Cost: " + batchCost / currentTrainingIndex);
        System.out.println("Train Accuracy: " + batchAccuracy / currentTrainingIndex * 100 + "%");
    }

    private static void reportPerformanceOnTest(NN NeuralNet, double[][] image, int[] answer) {
        double cost = 0;
        int correct = 0;
        for (int k = 63000; k < 70000; k++) {
            double[] expectedOutputs = new double[10];
            expectedOutputs[answer[k]] = 1.0;
            cost += NeuralNet.calculateCosts(image[k], expectedOutputs);
            int guess = 0;
            double[] guesses = NeuralNet.calculateOutput(image[k]);
            for (int i = 0; i < guesses.length; i++) {
                guess = (guesses[guess] > guesses[i]) ? guess : i;
            }
            correct += (guess == answer[k]) ? 1 : 0;
        }
        System.out.println("Test Avg java.Cost: " + cost / 7000);
        System.out.println("Test Accuracy: " + correct / 7000.0 * 100 + "%");
    }

    private static boolean evaluateOutput(double[] output, int answer) {
        int guess = 0;
        for (int j = 0; j < output.length; j++) {
            if (output[j] > output[guess]) guess = j;
        }
        return guess == answer;
    }
}
