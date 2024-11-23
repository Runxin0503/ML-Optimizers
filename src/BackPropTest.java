import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BackPropTest {

    /** Tests expected crossEntropy loss on 1 output node */
    @Test
    void crossEntropyLossTest(){
        double[] exampleData = new double[784];
        NN NeuralNet = new NN(Activation.ReLU,Activation.softmax,Cost.crossEntropy,784,1);
        assertEquals(0,NeuralNet.calculateCosts(exampleData,new double[]{1}));
        assertEquals(1_000_000,NeuralNet.calculateCosts(exampleData,new double[]{0}));
    }

    /** Tests expected crossEntropy loss on 10 output node */
    @Test
    void crossEntropyLossTest2(){
        double[] exampleData = new double[784];
        NN NeuralNet = new NN(Activation.ReLU,Activation.softmax,Cost.crossEntropy,784,10);
        double[] ones = new double[10];
        Arrays.fill(ones,1);
        assertAlmostEquals(-10*Math.log(0.9),NeuralNet.calculateCosts(exampleData,new double[10]));
        assertAlmostEquals(-10*Math.log(0.1),NeuralNet.calculateCosts(exampleData,ones));
    }

    /** Tests Activation Function none */
    @Test
    void noneAFTest() {
        //TODO implement
    }

    /** Tests Activation Function ReLU */
    @Test
    void ReLUTest() {
        //TODO implement
    }

    /** Tests Activation Function Sigmoid */
    @Test
    void SigmoidTest() {
        //TODO implement
    }

    /** Tests Activation Function Tanh */
    @Test
    void TanhTest() {
        //TODO implement
    }

    /** Tests Activation Function LeakyReLU */
    @Test
    void LeakyReLUTest() {
        //TODO implement
    }

    /** Tests Activation Function softmax */
    @Test
    void softmaxTest() {
        //TODO implement
    }

    private static void assertAlmostEquals(double expected, double actual){
        assertTrue(Math.abs(expected-actual)<0.00001);
    }
}
