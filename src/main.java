import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class main {

    public static void main(String[] args) throws Exception {
        double[][] image = new double[70000][784];
        int[] answer = new int[70000];
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader("lib/MNIST DATA.csv"))) {
            String line;
            int count=0;
            while ((line = bufferedReader.readLine()) != null) {
                String[] parts = line.split(",");
                answer[count]=Integer.parseInt(parts[0]);
                for(int i=1;i<parts.length;i++){
                    image[count][i-1]=Integer.parseInt(parts[i])/255.0;
                }
                count++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("LETS START LEARNING!\n--------------------");

        NN NeuralNet = new NN(Activation.ReLU,Activation.softmax,Cost.crossEntropy,784,100,10);

        final int batchSize = 10;
        double batchCost = 0;
        int batchAccuracy = 0;
        for (int trainingIndex = 0; trainingIndex < 6300; trainingIndex++) {
            double[][] trainBatchInputs = new double[10][784];
            double[][] trainBatchOutputs = new double[10][10];
            for (int i = 0; i < batchSize; i++) {
                trainBatchInputs[i] = image[i + trainingIndex * 10];
                trainBatchOutputs[i][answer[i + trainingIndex * 10]] = 1.0;
            }

            NN.learn(NeuralNet,1, 0.9, trainBatchInputs, trainBatchOutputs);
            for(int i=0;i< trainBatchInputs.length;i++){
                batchCost += NeuralNet.calculateCosts(trainBatchInputs[i], trainBatchOutputs[i]);
                int guess = 0;
                double[] guesses = NeuralNet.calculateOutput(trainBatchInputs[i]);
                for (int j = 0; j < guesses.length; j++) {
                    if(guesses[j] > guesses[guess]) guess = j;
                }
                batchAccuracy += (guess == answer[i + trainingIndex * 10]) ? 1 : 0;
            }

            if (trainingIndex % 700 == 0) {
                System.out.println("Iterations " + (trainingIndex * 10) + "~" + (trainingIndex*10+7000));
                reportPerformance(NeuralNet,image,answer);
                System.out.println("Train Avg Cost: " + batchCost / 7000);
                System.out.println("Train Accuracy: " + batchAccuracy / 7000 * 100 + "%\n--------------------");
                batchCost = 0;
                batchAccuracy = 0;
            }
        }
    }

    private static void reportPerformance(NN NeuralNet, double[][] image, int[] answer) {
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
        System.out.println("Test Avg Cost: " + cost / 7000);
        System.out.println("Test Accuracy: " + correct / 7000.0 * 100 + "%");
    }
}
