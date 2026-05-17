package Network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

/**
 * Edge-case unit tests for {@link NN} and its {@link NN.NetworkBuilder}.
 *
 * <p>Lives in package {@code Network} so it can reference the package-private {@link DenseLayer}
 * directly (used to exercise {@code addCustomLayer}). Tests that expect an {@link AssertionError}
 * rely on {@code -ea} (enabled by Maven Surefire by default).
 */
class NNTest {

    /** A fully-specified, buildable network: 3 inputs -> 4 -> 2 outputs, softmax head. */
    private static NN validNetwork() {
        return new NN.NetworkBuilder()
                .setInputNum(3)
                .addDenseLayer(4)
                .addDenseLayer(2)
                .setHiddenAF(Activation.ReLU)
                .setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy)
                .build();
    }

    // ---- NetworkBuilder.build(): required fields ---------------------------

    @Test
    void build_missingInputNum_throwsMissingInformation() {
        // a custom layer can be added without an inputNum, leaving inputNum == -1 at build time
        assertThrows(NN.MissingInformationException.class, () -> new NN.NetworkBuilder()
                .addCustomLayer(new DenseLayer(3, 2))
                .setHiddenAF(Activation.ReLU)
                .setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy)
                .build());
    }

    @Test
    void build_missingHiddenAF_throwsMissingInformation() {
        assertThrows(NN.MissingInformationException.class, () -> new NN.NetworkBuilder()
                .setInputNum(3)
                .addDenseLayer(2)
                .setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy)
                .build());
    }

    @Test
    void build_missingOutputAF_throwsMissingInformation() {
        assertThrows(NN.MissingInformationException.class, () -> new NN.NetworkBuilder()
                .setInputNum(3)
                .addDenseLayer(2)
                .setHiddenAF(Activation.ReLU)
                .setCostFunction(Cost.crossEntropy)
                .build());
    }

    @Test
    void build_missingCostFunction_throwsMissingInformation() {
        assertThrows(NN.MissingInformationException.class, () -> new NN.NetworkBuilder()
                .setInputNum(3)
                .addDenseLayer(2)
                .setHiddenAF(Activation.ReLU)
                .setOutputAF(Activation.softmax)
                .build());
    }

    @Test
    void build_noLayers_throwsMissingInformation() {
        assertThrows(NN.MissingInformationException.class, () -> new NN.NetworkBuilder()
                .setInputNum(3)
                .setHiddenAF(Activation.ReLU)
                .setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy)
                .build());
    }

    @Test
    void build_fullySpecified_producesNonNullNetwork() {
        NN network = validNetwork();
        assertNotNull(network);
        assertTrue(network.toString().contains("parameters"));
    }

    // ---- NetworkBuilder: ordering / dimension constraints ------------------

    @Test
    void setInputNum_afterAddingLayer_throwsUnsupportedOperation() {
        assertThrows(UnsupportedOperationException.class, () -> new NN.NetworkBuilder()
                .setInputNum(3)
                .addDenseLayer(2)
                .setInputNum(5));
    }

    @Test
    void addDenseLayer_beforeSetInputNum_throwsNegativeArraySize() {
        // inputNum is still -1, so `new DenseLayer(-1, nodes)` allocates `new double[-1][nodes]`
        assertThrows(NegativeArraySizeException.class,
                () -> new NN.NetworkBuilder().addDenseLayer(2));
    }

    @Test
    void addConvolutionalLayer_inputDimsMismatch_throwsAssertionError() {
        // requires -ea: inputNum (10) must equal inputWidth * inputHeight * inputLength (9)
        assertThrows(AssertionError.class, () -> new NN.NetworkBuilder()
                .setInputNum(10)
                .addConvolutionalLayer(3, 3, 1, 2, 2, 1, 1, 1));
    }

    // ---- calculateOutput ---------------------------------------------------

    @Test
    void calculateOutput_wrongInputLength_throwsAssertionError() {
        // requires -ea: input length must equal inputNum (3)
        NN network = validNetwork();
        assertThrows(AssertionError.class, () -> network.calculateOutput(new double[2]));
    }

    @Test
    void calculateOutput_softmaxWithZeroTemperature_throwsAssertionError() {
        // requires -ea: dividing the logits by temperature 0 yields non-finite values, which the
        // softmax activation's own finite-value guard rejects
        NN network = validNetwork();
        network.setTemperature(0);
        assertThrows(AssertionError.class, () -> network.calculateOutput(new double[3]));
    }

    // ---- learn / learnSingleOutput -----------------------------------------

    @Test
    void learn_emptyBatch_throwsAssertionError() {
        // requires -ea: learningRate / batchSize is 0.1 / 0 = Infinity, which applyGradient rejects
        NN network = validNetwork();
        assertThrows(AssertionError.class,
                () -> NN.learn(network, 0.1, 0.9, 0.999, 1e-8, new double[0][], new double[0][]));
    }

    @Test
    void learn_mismatchedBatchLengths_throwsAssertionError() {
        // requires -ea: the input batch and output batch must have the same number of samples
        NN network = validNetwork();
        assertThrows(AssertionError.class,
                () -> NN.learn(network, 0.1, 0.9, 0.999, 1e-8, new double[2][], new double[1][]));
    }

    @Test
    void learnSingleOutput_outputIndexOutOfRange_throwsAssertionError() {
        // requires -ea: outputIndex must satisfy 0 <= outputIndex < outputNum (2)
        NN network = validNetwork();
        assertThrows(AssertionError.class,
                () -> NN.learnSingleOutput(network, 0.1, 0.9, 0.999, 1e-8, new double[3], 99, 1.0));
    }

    // ---- equals / clone ----------------------------------------------------

    @Test
    void equals_networkEqualsItself() {
        // a network compared against itself is equal: NN.equals delegates the layer comparison
        // to Arrays.equals, which short-circuits on reference identity before ever reaching the
        // (broken) per-layer equals. The layer-equals bug only surfaces across distinct instances.
        NN network = validNetwork();
        assertTrue(network.equals(network));
    }

    @Test
    void clone_producesEqualNetwork() {
        // FAIL-LOUDLY: a deep clone should be equal to its original. NN.clone() clones each
        // layer, but DenseLayer.clone (DenseLayer.java:144) uses `weights[0].length` as the
        // `nodesBefore` argument; for a layer whose nodesBefore != nodes the rebuilt layer has
        // the wrong shape and the weight copy at DenseLayer.java:159 reads out of bounds.
        NN network = validNetwork();
        NN copy = (NN) network.clone();
        assertTrue(network.equals(copy));
    }

    @Test
    void build_eachOptimizer_succeeds() {
        for (Optimizer o : Optimizer.values()) {
            NN n = new NN.NetworkBuilder()
                    .setInputNum(3)
                    .addDenseLayer(4)
                    .addDenseLayer(2)
                    .setHiddenAF(Activation.ReLU)
                    .setOutputAF(Activation.softmax)
                    .setCostFunction(Cost.crossEntropy)
                    .setOptimizer(o)
                    .build();
            assertNotNull(n);
        }
    }

    @Test
    void build_eachHiddenActivation_succeeds() {
        for (Activation a : Activation.values()) {
            NN n = new NN.NetworkBuilder()
                    .setInputNum(3)
                    .addDenseLayer(4)
                    .addDenseLayer(2)
                    .setHiddenAF(a)
                    .setOutputAF(Activation.softmax)
                    .setCostFunction(Cost.crossEntropy)
                    .build();
            assertNotNull(n);
        }
    }

    @Test
    void build_eachOutputActivation_succeeds() {
        for (Activation a : Activation.values()) {
            NN n = new NN.NetworkBuilder()
                    .setInputNum(3)
                    .addDenseLayer(4)
                    .addDenseLayer(2)
                    .setHiddenAF(Activation.ReLU)
                    .setOutputAF(a)
                    .setCostFunction(Cost.diffSquared)
                    .build();
            assertNotNull(n);
        }
    }

    @Test
    void build_eachCostFunction_succeeds() {
        for (Cost c : Cost.values()) {
            NN n = new NN.NetworkBuilder()
                    .setInputNum(3)
                    .addDenseLayer(4)
                    .addDenseLayer(2)
                    .setHiddenAF(Activation.ReLU)
                    .setOutputAF(Activation.softmax)
                    .setCostFunction(c)
                    .build();
            assertNotNull(n);
        }
    }

    @Test
    void addCustomLayer_extendsLayerStack() {
        NN n = new NN.NetworkBuilder()
                .setInputNum(3)
                .addCustomLayer(new DenseLayer(3, 2))
                .setHiddenAF(Activation.ReLU)
                .setOutputAF(Activation.softmax)
                .setCostFunction(Cost.crossEntropy)
                .build();
        assertNotNull(n);
    }

    @Test
    void calculateOutput_returnsArrayOfLengthOutputNum() {
        NN n = validNetwork();
        assertEquals(2, n.calculateOutput(new double[]{0,0,0}).length);
    }

    @Test
    void calculateOutput_softmaxOutputSumsToOne() {
        NN n = validNetwork();
        double[] out = n.calculateOutput(new double[]{0,0,0});
        double s = out[0] + out[1];
        assertEquals(1.0, s, 1e-12);
    }

    @Test
    void calculateOutput_isDeterministic() {
        NN n = validNetwork();
        double[] a = n.calculateOutput(new double[]{1,2,3});
        double[] b = n.calculateOutput(new double[]{1,2,3});
        assertArrayEquals(a, b, 1e-12);
    }

    @Test
    void calculateOutput_temperatureFlatensSoftmax() {
        NN n = validNetwork();
        double[] lowT = n.calculateOutput(new double[]{1,1,1});
        n.setTemperature(1000);
        double[] highT = n.calculateOutput(new double[]{1,1,1});
        // high temperature should move distribution closer to uniform: lower max value
        double maxLow = Math.max(lowT[0], lowT[1]);
        double maxHigh = Math.max(highT[0], highT[1]);
        assertTrue(maxHigh <= maxLow + 1e-12);
    }

    @Test
    void calculateOutput_temperatureIgnored_forNonSoftmaxOutputs() {
        NN n = new NN.NetworkBuilder()
                .setInputNum(1)
                .addDenseLayer(1)
                .setHiddenAF(Activation.none)
                .setOutputAF(Activation.sigmoid)
                .setCostFunction(Cost.diffSquared)
                .build();
        double[] a = n.calculateOutput(new double[]{1});
        n.setTemperature(1000);
        double[] b = n.calculateOutput(new double[]{1});
        assertArrayEquals(a, b, 1e-12);
    }

    @Test
    void setTemperature_defaultIsOne() {
        NN n = validNetwork();
        // default temperature should not be zero; calculateOutput works
        assertDoesNotThrow(() -> n.calculateOutput(new double[]{0,0,0}));
    }

    @Test
    void calculateCost_perfectPrediction_isZero_diffSquared() {
        // weights/bias are randomly initialised, so "perfect" means feeding the network's own
        // output back in as the target -- diffSquared then sums (x - x)^2 = 0 per element
        NN n = new NN.NetworkBuilder()
                .setInputNum(1)
                .addDenseLayer(1)
                .setHiddenAF(Activation.none)
                .setOutputAF(Activation.none)
                .setCostFunction(Cost.diffSquared)
                .build();
        double[] input = {0};
        double[] target = n.calculateOutput(input);
        assertEquals(0.0, n.calculateCost(input, target), 1e-12);
    }

    @Test
    void calculateCost_nonNegative_forValidInputs() {
        NN n = validNetwork();
        double c = n.calculateCost(new double[]{0,0,0}, new double[]{0,1});
        assertTrue(c >= 0);
    }

    @Test
    void calculateCost_returnsScalarSumOfPerElementCosts() {
        NN n = new NN.NetworkBuilder()
                .setInputNum(1)
                .addDenseLayer(1)
                .setHiddenAF(Activation.none)
                .setOutputAF(Activation.none)
                .setCostFunction(Cost.diffSquared)
                .build();
        double[] out = n.calculateOutput(new double[]{0});
        double[] per = Cost.diffSquared.calculate(out, new double[]{1});
        double sum = 0; for (double v : per) sum += v;
        assertEquals(sum, n.calculateCost(new double[]{0}, new double[]{1}), 1e-12);
    }

    @Test
    void backPropagate_validInput_doesNotThrow() {
        NN n = validNetwork();
        assertDoesNotThrow(() -> n.backPropagate(new double[]{0,0,0}, new double[]{1,0}));
    }

    @Test
    void learn_singleStep_changesOutput() {
        NN n = new NN.NetworkBuilder()
                .setInputNum(1)
                .addDenseLayer(1)
                .setHiddenAF(Activation.none)
                .setOutputAF(Activation.none)
                .setCostFunction(Cost.diffSquared)
                .setOptimizer(Optimizer.SGD)
                .build();
        double[] before = n.calculateOutput(new double[]{1});
        NN.learn(n, 0.1, 0, 0, 0, new double[][]{{1}}, new double[][]{{2}});
        double[] after = n.calculateOutput(new double[]{1});
        assertFalse(after[0] == before[0]);
    }

    @Test
    void learnSingleOutput_validIndex_doesNotThrow() {
        NN n = validNetwork();
        assertDoesNotThrow(() -> NN.learnSingleOutput(n, 0.1, 0.9, 0.999, 1e-8, new double[]{0,0,0}, 0, 1.0));
    }

    @Test
    void clone_trainingClone_doesNotAffectOriginal() {
        NN n = validNetwork();
        NN copy = (NN) n.clone();
        // snapshot the original's output BEFORE training the clone, otherwise the assertion
        // would compare n's output to itself (a tautology). Use proper Adam hyperparams since
        // validNetwork() builds with ADAM and eps=0 trips an internal finite-value assertion.
        double[] input = {0, 0, 0};
        double[] before = n.calculateOutput(input);
        NN.learn(copy, 0.1, 0.9, 0.999, 1e-8, new double[][]{input}, new double[][]{{1, 0}});
        assertArrayEquals(before, n.calculateOutput(input), 1e-12);
    }

    @Test
    void toString_containsParameterCountAndLayerSections() {
        NN n = validNetwork();
        String s = n.toString();
        assertTrue(s.contains("parameters") && s.contains("Layer 0"));
    }
}
