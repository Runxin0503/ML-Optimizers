import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class main {
    public static void main(String[] args) throws Exception {
        double[][] holy = new double[70000][784];
        int[] moly = new int[70000];
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader("lib/MNIST DATA.csv"))) {
            String line;
            int count=0;
            while ((line = bufferedReader.readLine()) != null) {
                String[] parts = line.split(",");
                moly[count]=Integer.parseInt(parts[0]);
                for(int i=1;i<parts.length;i++){
                    holy[count][i-1]=Integer.parseInt(parts[i])/255.0;
                }
                count++;
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println("LETS START LEARNING!\n--------------------");
        NN iCanREAD = new NN(Activation.ReLU,Activation.softmax,Cost.crossEntropy,784,100,100,10);
        for(int iteration=0;iteration<2;iteration++) {
            for (int l = 0; l < 6300; l++) {
                double[][] testCaseInput = new double[10][784];
                double[][] testCaseOutput = new double[10][10];
                for (int i = 0; i < 10; i++) {
                    testCaseInput[i] = holy[i + l * 10];
                    testCaseOutput[i][moly[i + l * 10]] = 1.0;
                }

                iCanREAD.learn(2, 0.9, testCaseInput, testCaseOutput);

                if (l % 700 == 0) {
                    double temp = 0;
                    int correct = 0;
                    for (int k = 63000; k < 70000; k++) {
                        double[] expectedOutputs = new double[10];
                        expectedOutputs[moly[k]] = 1.0;
                        temp += iCanREAD.calculateCosts(holy[k], expectedOutputs);
                        int guess = 0;
                        double[] guesses = iCanREAD.calculateOutput(holy[k]);
                        for (int i = 0; i < guesses.length; i++) {
                            guess = (guesses[guess] > guesses[i]) ? guess : i;
                        }
                        correct += (guess == moly[k]) ? 1 : 0;
                    }
                    double traincorrect = 0;
                    for (int k = 0; k < l * 10; k++) {
                        double[] expectedOutputs = new double[10];
                        expectedOutputs[moly[k]] = 1.0;
                        int guess = 0;
                        double[] guesses = iCanREAD.calculateOutput(holy[k]);
                        for (int i = 0; i < guesses.length; i++) {
                            guess = (guesses[guess] > guesses[i]) ? guess : i;
                        }
                        traincorrect += (guess == moly[k]) ? 1 : 0;
                    }
                    System.out.println("Iteration "+(iteration+1));
                    System.out.print("Avg Cost(" + (l * 10 + 7000) + "): ");
                    System.out.println(Math.floor(temp / 70) / 100.0);
                    System.out.println("Test Accuracy: " + Math.floor(correct / 70.0 * 100) / 100.0 + "%");
                    System.out.println("Train Accuracy: " + Math.floor(traincorrect / l * 1000) / 100 + "%\n--------------------");
                }
            }
        }
        // int[] listoftargets = {824,7532,8642,15345,4233,59725,9172,19837,54326,8253};
        // for(int l=0;l<listoftargets.length;l++){
        //     int guess=0;
        //     double[] guesses = iCanREAD.calculateOutput(holy[listoftargets[l]]);
        //     for(int i=0;i<guesses.length;i++){
        //         guess=guesses[guess] > guesses[i]?guess:i;
        //     }
        //     System.out.println((guess==moly[listoftargets[l]])?"Correct!":("Correct Answer: " + moly[listoftargets[l]] + " --- Your Answer: " + guess));
        // }
    }
}
