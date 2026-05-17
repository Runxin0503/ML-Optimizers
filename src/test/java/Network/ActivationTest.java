package Network;

import org.junit.jupiter.api.Test;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Edge-case unit tests for {@link Activation}.
 *
 * <p>Lives in package {@code Network} so it can call the package-private {@code calculate},
 * {@code derivative}, and {@code getInitializer} members directly. Tests that expect an
 * {@link AssertionError} rely on {@code -ea} (enabled by Maven Surefire by default).
 */
class ActivationTest {

    private static final double DELTA = 1e-9;

    // ---- none --------------------------------------------------------------

    @Test
    void none_calculate_isIdentity() {
        assertArrayEquals(new double[]{1, -2, 3},
                Activation.none.calculate(new double[]{1, -2, 3}), DELTA);
    }

    @Test
    void none_derivative_passesGradientThrough() {
        assertArrayEquals(new double[]{5, 6, 7},
                Activation.none.derivative(new double[]{0, 0, 0}, new double[]{5, 6, 7}), DELTA);
    }

    // ---- ReLU --------------------------------------------------------------

    @Test
    void relu_calculate_clampsAtZeroIncludingBoundary() {
        // boundary: input 0 is not > 0, so it maps to 0
        assertArrayEquals(new double[]{0, 0, 2},
                Activation.ReLU.calculate(new double[]{-1, 0, 2}), DELTA);
    }

    @Test
    void relu_derivative_isStepFunctionWithZeroAtBoundary() {
        assertArrayEquals(new double[]{0, 0, 10},
                Activation.ReLU.derivative(new double[]{-1, 0, 2}, new double[]{10, 10, 10}), DELTA);
    }

    // ---- sigmoid -----------------------------------------------------------

    @Test
    void sigmoid_calculate_atZeroIsHalf() {
        assertArrayEquals(new double[]{0.5}, Activation.sigmoid.calculate(new double[]{0}), DELTA);
    }

    @Test
    void sigmoid_calculate_saturatesForLargeMagnitude() {
        assertEquals(1.0, Activation.sigmoid.calculate(new double[]{100})[0], 1e-10);
        assertEquals(0.0, Activation.sigmoid.calculate(new double[]{-100})[0], 1e-10);
    }

    @Test
    void sigmoid_derivative_atZeroIsQuarter() {
        assertArrayEquals(new double[]{0.25},
                Activation.sigmoid.derivative(new double[]{0}, new double[]{1}), DELTA);
    }

    @Test
    void sigmoid_derivative_saturatedIsZeroAndFinite() {
        // vanishing gradient at saturation, but the value stays finite (no AssertionError)
        assertEquals(0.0, Activation.sigmoid.derivative(new double[]{100}, new double[]{1})[0], 1e-10);
    }

    // ---- tanh --------------------------------------------------------------

    @Test
    void tanh_calculate_atZeroIsZero() {
        assertArrayEquals(new double[]{0.0}, Activation.tanh.calculate(new double[]{0}), DELTA);
    }

    @Test
    void tanh_calculate_saturatesForLargeMagnitude() {
        assertEquals(1.0, Activation.tanh.calculate(new double[]{100})[0], DELTA);
        assertEquals(-1.0, Activation.tanh.calculate(new double[]{-100})[0], DELTA);
    }

    @Test
    void tanh_derivative_atZeroIsGradient() {
        assertArrayEquals(new double[]{1.0},
                Activation.tanh.derivative(new double[]{0}, new double[]{1}), DELTA);
    }

    // ---- LeakyReLU ---------------------------------------------------------

    @Test
    void leakyRelu_calculate_leaksNegativesAndZerosBoundary() {
        // boundary: input 0 is not > 0, so it maps to 0.1 * 0 = 0
        assertArrayEquals(new double[]{-1.0, 0.0, 5.0},
                Activation.LeakyReLU.calculate(new double[]{-10, 0, 5}), DELTA);
    }

    @Test
    void leakyRelu_derivative_isOneOrTenthSlope() {
        assertArrayEquals(new double[]{1.0, 1.0, 10.0},
                Activation.LeakyReLU.derivative(new double[]{-10, 0, 5}, new double[]{10, 10, 10}), DELTA);
    }

    // ---- softmax -----------------------------------------------------------

    @Test
    void softmax_singleElement_isOne() {
        assertArrayEquals(new double[]{1.0}, Activation.softmax.calculate(new double[]{5}), DELTA);
    }

    @Test
    void softmax_equalLogits_isUniform() {
        assertArrayEquals(new double[]{1.0 / 3, 1.0 / 3, 1.0 / 3},
                Activation.softmax.calculate(new double[]{2, 2, 2}), DELTA);
    }

    @Test
    void softmax_largePositiveSpread_isNumericallyStable() {
        // the max-shift trick keeps this finite and a valid distribution
        double[] result = Activation.softmax.calculate(new double[]{0, 1000});
        assertArrayEquals(new double[]{0.0, 1.0}, result, DELTA);
    }

    @Test
    void softmax_derivative_usesJacobianForm() {
        // softmax([0,0]) = [0.5, 0.5]; with gradient [1,0], dot = 0.5
        // out[i] = softmax[i] * (gradient[i] - dot)
        assertArrayEquals(new double[]{0.25, -0.25},
                Activation.softmax.derivative(new double[]{0, 0}, new double[]{1, 0}), DELTA);
    }

    @Test
    void softmax_allNegativeLogits_shouldReturnValidDistribution() {
        // FAIL-LOUDLY: Activation.java:99 seeds `max` with Double.MIN_VALUE (the smallest *positive*
        // double) instead of a very negative value. For all-negative logits every exp() underflows
        // to 0, the denominator becomes 0, and the result is NaN -> the internal finite-check throws.
        // The softmax of any finite input should be a valid probability distribution.
        double[] result = Activation.softmax.calculate(new double[]{-1000, -2000});
        assertTrue(Double.isFinite(result[0]) && Double.isFinite(result[1]),
                "softmax produced non-finite output for all-negative logits");
        assertEquals(1.0, result[0] + result[1], 1e-9, "softmax output must sum to 1");
    }

    // ---- contract: non-finite inputs --------------------------------------

    @Test
    void calculate_withNaNInput_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Activation.ReLU.calculate(new double[]{Double.NaN}));
    }

    @Test
    void calculate_withInfiniteInput_throwsAssertionError() {
        // requires -ea; the input guard fires before the function is applied
        assertThrows(AssertionError.class,
                () -> Activation.tanh.calculate(new double[]{Double.POSITIVE_INFINITY}));
    }

    @Test
    void derivative_withNaNGradient_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Activation.ReLU.derivative(new double[]{0.0}, new double[]{Double.NaN}));
    }

    // ---- getInitializer ----------------------------------------------------

    @Test
    void getInitializer_producesFiniteWeightsForNormalSizes() {
        for (Activation af : Activation.values()) {
            Supplier<Double> init = Activation.getInitializer(af, 8, 8);
            assertNotNull(init, "initializer for " + af + " should not be null");
            assertTrue(Double.isFinite(init.get()),
                    "initializer for " + af + " produced a non-finite weight");
        }
    }

    @Test
    void getInitializer_zeroSizes_producesNonFiniteOrThrows() {
        // He init uses sqrt(2 / (inputNum + outputNum)); with both sizes 0 the std-dev is infinite,
        // so the generated weight is non-finite. The exact behaviour of nextGaussian with an
        // infinite std-dev is JDK-dependent, so accept either a non-finite draw or an exception.
        Supplier<Double> init = Activation.getInitializer(Activation.ReLU, 0, 0);
        try {
            assertFalse(Double.isFinite(init.get()));
        } catch (RuntimeException acceptable) {
            // also acceptable
        }
    }
}
