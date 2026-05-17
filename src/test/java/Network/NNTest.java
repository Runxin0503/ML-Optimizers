package Network;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
}
