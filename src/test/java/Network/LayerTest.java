package Network;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Edge-case unit tests for the abstract base {@link Layer}.
 *
 * <p>Exercised through {@link StubLayer}, a minimal in-package concrete subclass with trivial
 * abstract-method bodies. Being in package {@code Network} lets the tests read and write
 * {@link Layer}'s {@code protected} fields directly, so the optimizer math can be checked with
 * exact, hand-set state.
 */
class LayerTest {

    private static final double DELTA = 1e-12;

    /** Minimal concrete {@link Layer} used purely to test the shared base-class behaviour. */
    private static final class StubLayer extends Layer {
        StubLayer(int nodes) {
            super(nodes);
        }

        @Override
        double[] calculateWeightedOutput(double[] input) {
            return input.clone();
        }

        @Override
        double[] updateGradient(double[] dz_dC, double[] x) {
            return dz_dC.clone();
        }

        @Override
        void clearGradient() {
            Arrays.fill(biasGradient, 0);
        }

        @Override
        public String toString() {
            return "StubLayer(" + nodes + ")";
        }

        @Override
        public Object clone() {
            return new StubLayer(nodes);
        }
    }

    // ---- constructor -------------------------------------------------------

    @Test
    void constructor_allocatesZeroedBiasAndGradient_velocitiesNull() {
        StubLayer layer = new StubLayer(3);
        assertEquals(3, layer.nodes);
        assertArrayZero(layer.bias);
        assertArrayZero(layer.biasGradient);
        assertNull(layer.biasVelocity);
        assertNull(layer.biasVelocitySquared);
        assertEquals(1, layer.t);
    }

    // ---- initialize: velocity-array allocation per optimizer ---------------

    @Test
    void initialize_sgd_allocatesNoVelocityArrays() {
        StubLayer layer = new StubLayer(2);
        layer.initialize(() -> 0.5, Optimizer.SGD);
        assertNull(layer.biasVelocity);
        assertNull(layer.biasVelocitySquared);
        assertEquals(0.5, layer.bias[0], DELTA);
        assertEquals(0.5, layer.bias[1], DELTA);
    }

    @Test
    void initialize_sgdMomentum_allocatesVelocityOnly() {
        StubLayer layer = new StubLayer(2);
        layer.initialize(() -> 0.0, Optimizer.SGD_MOMENTUM);
        assertNotNull(layer.biasVelocity);
        assertEquals(2, layer.biasVelocity.length);
        assertNull(layer.biasVelocitySquared);
    }

    @Test
    void initialize_rmsProp_allocatesVelocitySquaredOnly() {
        StubLayer layer = new StubLayer(2);
        layer.initialize(() -> 0.0, Optimizer.RMS_PROP);
        assertNull(layer.biasVelocity);
        assertNotNull(layer.biasVelocitySquared);
        assertEquals(2, layer.biasVelocitySquared.length);
    }

    @Test
    void initialize_adam_allocatesBothVelocityArrays() {
        StubLayer layer = new StubLayer(2);
        layer.initialize(() -> 0.0, Optimizer.ADAM);
        assertNotNull(layer.biasVelocity);
        assertNotNull(layer.biasVelocitySquared);
    }

    // ---- applyGradient: exact optimizer math -------------------------------

    @Test
    void applyGradient_sgd_subtractsScaledGradient() {
        StubLayer layer = new StubLayer(2);
        layer.bias[0] = 1.0;
        layer.bias[1] = 2.0;
        layer.biasGradient[0] = 0.5;
        layer.biasGradient[1] = -1.0;

        layer.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);

        assertEquals(1.0 - 0.1 * 0.5, layer.bias[0], DELTA);
        assertEquals(2.0 - 0.1 * -1.0, layer.bias[1], DELTA);
    }

    @Test
    void applyGradient_sgdMomentum_updatesVelocityThenBias() {
        StubLayer layer = new StubLayer(1);
        layer.bias[0] = 1.0;
        layer.biasGradient[0] = 2.0;
        layer.biasVelocity = new double[]{0.0};

        layer.applyGradient(Optimizer.SGD_MOMENTUM, 0.1, 0.9, 0, 0);

        // velocity = 0.9 * 0 + (1 - 0.9) * 2.0 = 0.2
        assertEquals(0.2, layer.biasVelocity[0], DELTA);
        // bias = 1.0 - 0.1 * 0.2 = 0.98
        assertEquals(0.98, layer.bias[0], DELTA);
    }

    @Test
    void applyGradient_rmsProp_updatesVelocitySquaredThenBias() {
        StubLayer layer = new StubLayer(1);
        layer.bias[0] = 1.0;
        layer.biasGradient[0] = 2.0;
        layer.biasVelocitySquared = new double[]{0.0};

        layer.applyGradient(Optimizer.RMS_PROP, 0.1, 0, 0.9, 1e-8);

        // velocitySquared = 0.9 * 0 + (1 - 0.9) * 2.0^2 = 0.4
        assertEquals(0.4, layer.biasVelocitySquared[0], DELTA);
        // bias = 1.0 - 0.1 * 2.0 / sqrt(0.4 + 1e-8)
        assertEquals(1.0 - 0.1 * 2.0 / Math.sqrt(0.4 + 1e-8), layer.bias[0], DELTA);
    }

    @Test
    void applyGradient_adam_accumulatesVelocities() {
        StubLayer layer = new StubLayer(1);
        layer.bias[0] = 1.0;
        layer.biasGradient[0] = 2.0;
        layer.biasVelocity = new double[]{0.0};
        layer.biasVelocitySquared = new double[]{0.0};

        layer.applyGradient(Optimizer.ADAM, 0.1, 0.9, 0.999, 1e-8);

        // the raw moment accumulation is correct
        assertEquals(0.2, layer.biasVelocity[0], DELTA);          // 0.9*0 + 0.1*2.0
        assertEquals(0.004, layer.biasVelocitySquared[0], DELTA); // 0.999*0 + 0.001*4.0
    }

    @Test
    void applyGradient_adam_appliesBiasCorrectedUpdate() {
        StubLayer layer = new StubLayer(1);
        layer.bias[0] = 1.0;
        layer.biasGradient[0] = 2.0;
        layer.biasVelocity = new double[]{0.0};
        layer.biasVelocitySquared = new double[]{0.0};

        layer.applyGradient(Optimizer.ADAM, 0.1, 0.9, 0.999, 1e-8);

        // FAIL-LOUDLY: with t = 1, the Adam bias-correction divides the moments by
        // (1 - momentum^t) and (1 - beta^t). Layer.java:93-94 instead *multiplies* by those
        // factors (Linalg.scale), where DenseLayer.java:95-96 correctly divides.
        // correct: m_hat = 0.2 / (1 - 0.9) = 2.0 ; v_hat = 0.004 / (1 - 0.999) = 4.0
        double mHat = 0.2 / (1 - 0.9);
        double vHat = 0.004 / (1 - 0.999);
        assertEquals(1.0 - 0.1 * mHat / Math.sqrt(vHat + 1e-8), layer.bias[0], 1e-9);
    }

    @Test
    void applyGradient_incrementsTimestepOnlyForAdam() {
        StubLayer sgd = new StubLayer(1);
        sgd.applyGradient(Optimizer.SGD, 0.1, 0, 0, 0);
        assertEquals(1, sgd.t, "SGD must not advance the Adam timestep");

        StubLayer adam = new StubLayer(1);
        adam.biasVelocity = new double[]{0.0};
        adam.biasVelocitySquared = new double[]{0.0};
        adam.applyGradient(Optimizer.ADAM, 0.1, 0.9, 0.999, 1e-8);
        assertEquals(2, adam.t, "ADAM must advance the timestep by one");
    }

    // ---- getNumParameters --------------------------------------------------

    @Test
    void getNumParameters_isBiasLength() {
        assertEquals(5, new StubLayer(5).getNumParameters());
        assertEquals(0, new StubLayer(0).getNumParameters());
    }

    // ---- equals ------------------------------------------------------------

    @Test
    void equals_identicalLayersAreEqual() {
        assertTrue(new StubLayer(2).equals(new StubLayer(2)));
    }

    @Test
    void equals_differentNodeCountsAreNotEqual() {
        assertFalse(new StubLayer(2).equals(new StubLayer(3)));
    }

    @Test
    void equals_differentBiasOrGradientAreNotEqual() {
        StubLayer a = new StubLayer(1);
        StubLayer b = new StubLayer(1);
        b.bias[0] = 1.0;
        assertFalse(a.equals(b));

        StubLayer c = new StubLayer(1);
        c.biasGradient[0] = 1.0;
        assertFalse(a.equals(c));
    }

    @Test
    void equals_nullVersusAllocatedVelocityAreNotEqual() {
        StubLayer a = new StubLayer(1);                 // biasVelocity == null
        StubLayer b = new StubLayer(1);
        b.biasVelocity = new double[1];                 // allocated
        assertFalse(a.equals(b));
    }

    // ---- helpers -----------------------------------------------------------

    private static void assertArrayZero(double[] array) {
        for (double v : array) assertEquals(0.0, v, DELTA);
    }
}
