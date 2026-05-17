package Network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

/**
 * Edge-case unit tests for {@link DenseLayer} (a package-private class, hence the {@code Network}
 * package declaration).
 *
 * <p>{@code weights} is {@code private}, so it cannot be read even from in-package tests. The
 * suite stays deterministic by using freshly-constructed layers (all-zero weights and bias) and
 * by driving weight state through {@code updateGradient} -> {@code applyGradient}, observing the
 * effect through {@code calculateWeightedOutput}. Tests that expect an {@link AssertionError}
 * rely on {@code -ea} (enabled by Maven Surefire by default).
 */
class DenseLayerTest {

    private static final double DELTA = 1e-12;

    // ---- constructor / getNumParameters ------------------------------------

    @Test
    void getNumParameters_isWeightsPlusBiases() {
        // nodesBefore * nodes weights + nodes biases
        assertEquals(2 * 3 + 3, new DenseLayer(2, 3).getNumParameters());
    }

    // ---- calculateWeightedOutput ------------------------------------------

    @Test
    void calculateWeightedOutput_freshLayerProducesZeroVectorOfOutputSize() {
        // before initialize(), weights and bias are all zero
        double[] output = new DenseLayer(2, 3).calculateWeightedOutput(new double[]{1, 2});
        assertArrayEquals(new double[]{0, 0, 0}, output, DELTA);
    }

    @Test
    void calculateWeightedOutput_wrongInputLength_throwsAssertionError() {
        // requires -ea: input length must equal nodesBefore (2 here)
        DenseLayer layer = new DenseLayer(2, 3);
        assertThrows(AssertionError.class, () -> layer.calculateWeightedOutput(new double[]{1, 2, 3}));
    }

    // ---- updateGradient ----------------------------------------------------

    @Test
    void updateGradient_returnsArrayOfNodesBeforeLength() {
        // fresh weights are zero, so the returned da_dC is a zero vector of length nodesBefore
        double[] daDC = new DenseLayer(2, 3).updateGradient(new double[]{1, 1, 1}, new double[]{1, 1});
        assertArrayEquals(new double[]{0, 0}, daDC, DELTA);
    }

    @Test
    void updateGradient_accumulatesIntoBiasGradient() {
        DenseLayer layer = new DenseLayer(1, 2);
        layer.updateGradient(new double[]{3, 4}, new double[]{1});
        assertArrayEquals(new double[]{3, 4}, layer.biasGradient, DELTA);

        // calling again without clearGradient accumulates rather than overwrites
        layer.updateGradient(new double[]{3, 4}, new double[]{1});
        assertArrayEquals(new double[]{6, 8}, layer.biasGradient, DELTA);
    }

    @Test
    void clearGradient_resetsBiasGradient() {
        DenseLayer layer = new DenseLayer(1, 2);
        layer.updateGradient(new double[]{3, 4}, new double[]{1});
        layer.clearGradient();
        assertArrayEquals(new double[]{0, 0}, layer.biasGradient, DELTA);
    }

    // ---- applyGradient: deterministic weight/bias update -------------------

    @Test
    void applyGradient_sgd_updatesWeightsAndBiasObservably() {
        // fresh 1->1 layer: weights = [[0]], bias = [0]
        DenseLayer layer = new DenseLayer(1, 1);
        layer.updateGradient(new double[]{3.0}, new double[]{2.0});
        // weightsGradient[0][0] = x*dz_dC = 2.0*3.0 = 6.0 ; biasGradient[0] = 3.0
        layer.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);
        // weights[0][0] = 0 - 0.1*6.0 = -0.6 ; bias[0] = 0 - 0.1*3.0 = -0.3
        // calculateWeightedOutput([1.0]) = (-0.6 * 1.0) + (-0.3) = -0.9
        assertArrayEquals(new double[]{-0.9}, layer.calculateWeightedOutput(new double[]{1.0}), DELTA);
    }

    @Test
    void applyGradient_accumulatedGradientDoublesWeightEffect() {
        DenseLayer layer = new DenseLayer(1, 1);
        layer.updateGradient(new double[]{3.0}, new double[]{2.0});
        layer.updateGradient(new double[]{3.0}, new double[]{2.0}); // accumulates: grad now doubled
        layer.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);
        // double of the single-update case (-0.9) -> -1.8
        assertArrayEquals(new double[]{-1.8}, layer.calculateWeightedOutput(new double[]{1.0}), DELTA);
    }

    @Test
    void clearGradient_makesSubsequentApplyGradientANoOp() {
        DenseLayer layer = new DenseLayer(1, 1);
        layer.updateGradient(new double[]{3.0}, new double[]{2.0});
        layer.clearGradient();
        layer.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);
        // gradients were zeroed, so weights and bias remain zero
        assertArrayEquals(new double[]{0.0}, layer.calculateWeightedOutput(new double[]{5.0}), DELTA);
    }

    @Test
    void initialize_thenApplyGradient_worksForEveryOptimizer() {
        // initialize() must allocate the weight-velocity arrays each optimizer needs; if it did
        // not, applyGradient would throw NullPointerException dereferencing them.
        for (Optimizer optimizer : new Optimizer[]{
                Optimizer.SGD, Optimizer.SGD_MOMENTUM, Optimizer.RMS_PROP, Optimizer.ADAM}) {
            DenseLayer layer = new DenseLayer(1, 1);
            layer.initialize(() -> 0.0, optimizer);
            layer.updateGradient(new double[]{1.0}, new double[]{1.0});
            assertDoesNotThrow(() -> layer.applyGradient(optimizer, 0.01, 0.9, 0.999, 1e-8),
                    "applyGradient failed for optimizer " + optimizer);
        }
    }

    // ---- equals ------------------------------------------------------------

    @Test
    void equals_layerEqualsItself() {
        // FAIL-LOUDLY: DenseLayer.java:135 reads `... || super.equals(obj)` (a `!` is missing),
        // so when the base fields DO match it returns false. A layer is therefore never equal
        // to itself or to an identical layer.
        DenseLayer layer = new DenseLayer(2, 3);
        assertTrue(layer.equals(layer));
    }

    // ---- clone -------------------------------------------------------------

    @Test
    void clone_squareLayer_behavesIdenticallyToOriginal() {
        DenseLayer original = new DenseLayer(2, 2);
        original.updateGradient(new double[]{1, 1}, new double[]{1, 1});
        original.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);

        DenseLayer copy = (DenseLayer) original.clone();
        double[] input = {1, 1};
        assertArrayEquals(original.calculateWeightedOutput(input),
                copy.calculateWeightedOutput(input), DELTA);
    }

    @Test
    void clone_nonSquareLayer_behavesIdenticallyToOriginal() {
        // FAIL-LOUDLY: DenseLayer.java:144 uses `weights[0].length` (which is `nodes`) as the
        // `nodesBefore` argument when rebuilding the layer, so a clone of a layer whose
        // nodesBefore != nodes has the wrong shape. Here the clone ends up shaped 3x3 instead
        // of 2x3 and rejects the 2-element input that the original accepts.
        DenseLayer original = new DenseLayer(2, 3);
        DenseLayer copy = (DenseLayer) original.clone();
        double[] input = {1, 1};
        assertArrayEquals(original.calculateWeightedOutput(input),
                copy.calculateWeightedOutput(input), DELTA);
    }

    // ---- toString ----------------------------------------------------------

    @Test
    void toString_mentionsBiases() {
        assertTrue(new DenseLayer(1, 1).toString().contains("Biases"));
    }

    @Test
    void constructor_oneByOne_works() {
        DenseLayer d = new DenseLayer(1, 1);
        assertEquals(2, d.getNumParameters());
    }

    @Test
    void constructor_zeroNodesBefore_yieldsEmptyWeightsMatrix() {
        // weights is `new double[0][3]`; getNumParameters returns 0 * 3 weights + 3 biases.
        // (Calling calculateWeightedOutput would still trip the Linalg.matrixMultiply
        // empty-supplier bug exercised separately by LinalgTest.)
        assertEquals(3, new DenseLayer(0, 3).getNumParameters());
    }

    @Test
    void getNumParameters_variantShapes() {
        assertEquals(5 * 5 + 5, new DenseLayer(5, 5).getNumParameters());
        assertEquals(5 * 1 + 1, new DenseLayer(5, 1).getNumParameters());
        assertEquals(1 * 5 + 5, new DenseLayer(1, 5).getNumParameters());
    }

    @Test
    void calculateWeightedOutput_returnsNewArray_notAlias() {
        DenseLayer d = new DenseLayer(2, 2);
        double[] in = {1, 1};
        double[] out = d.calculateWeightedOutput(in);
        assertNotSame(in, out);
    }

    @Test
    void calculateWeightedOutput_returnsArrayOfLengthNodes() {
        DenseLayer d = new DenseLayer(3, 4);
        assertEquals(4, d.calculateWeightedOutput(new double[]{1,1,1}).length);
    }

    @Test
    void updateGradient_zeroDzDc_leavesGradientsUnchanged() {
        DenseLayer d = new DenseLayer(2, 2);
        d.updateGradient(new double[]{0,0}, new double[]{1,1});
        assertArrayEquals(new double[]{0,0}, d.biasGradient, DELTA);
    }

    @Test
    void updateGradient_nanDzDc_propagatesSilently() {
        DenseLayer d = new DenseLayer(1, 2);
        d.updateGradient(new double[]{Double.NaN, 1.0}, new double[]{1});
        assertTrue(Double.isNaN(d.biasGradient[0]));
    }

    @Test
    void updateGradient_returnsNewArray_notAlias() {
        DenseLayer d = new DenseLayer(2, 2);
        double[] dz = new double[]{1,2};
        double[] ret = d.updateGradient(dz, new double[]{1,1});
        assertNotSame(dz, ret);
    }

    @Test
    void updateGradient_xShorterThanNodesBefore_throwsArrayIndexOutOfBounds() {
        DenseLayer d = new DenseLayer(3, 2);
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> d.updateGradient(new double[]{1,2}, new double[]{1,2}));
    }

    @Test
    void clearGradient_calledTwice_isStillANoOp() {
        DenseLayer d = new DenseLayer(1,1);
        d.updateGradient(new double[]{1}, new double[]{1});
        d.clearGradient();
        d.clearGradient();
        d.applyGradient(Optimizer.SGD, 0.1, 0,0,0);
        assertArrayEquals(new double[]{0.0}, d.calculateWeightedOutput(new double[]{1.0}), DELTA);
    }

    @Test
    void equals_sameShapeDifferentWeights_areNotEqual() {
        DenseLayer a = new DenseLayer(1,1);
        DenseLayer b = new DenseLayer(1,1);
        b.updateGradient(new double[]{1.0}, new double[]{1.0});
        b.applyGradient(Optimizer.SGD, 0.1, 0,0,0);
        assertFalse(a.equals(b));
    }

    @Test
    void clone_independence_modifyingCloneDoesNotAffectOriginal() {
        DenseLayer orig = new DenseLayer(1,1);
        DenseLayer copy = (DenseLayer) orig.clone();
        copy.updateGradient(new double[]{1.0}, new double[]{1.0});
        copy.applyGradient(Optimizer.SGD, 0.1, 0,0,0);
        // original remains unchanged
        assertArrayEquals(orig.calculateWeightedOutput(new double[]{1.0}),
                new double[]{0.0}, DELTA);
    }
}
