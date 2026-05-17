package Network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;

/**
 * Edge-case unit tests for {@link Cost}.
 *
 * <p>Lives in package {@code Network} so it can call the package-private {@code calculate} and
 * {@code derivative} members directly. Tests that expect an {@link AssertionError} rely on
 * {@code -ea} (enabled by Maven Surefire by default).
 */
class CostTest {

    private static final double DELTA = 1e-9;

    // ---- diffSquared (MSE) -------------------------------------------------

    @Test
    void diffSquared_calculate_knownResult() {
        // (x - y)^2 / n  per element
        assertArrayEquals(new double[]{4.0 / 2, 0.0},
                Cost.diffSquared.calculate(new double[]{3, 1}, new double[]{1, 1}), DELTA);
    }

    @Test
    void diffSquared_calculate_perfectPredictionIsZero() {
        assertArrayEquals(new double[]{0, 0},
                Cost.diffSquared.calculate(new double[]{5, 5}, new double[]{5, 5}), DELTA);
    }

    @Test
    void diffSquared_calculate_singleElement() {
        assertArrayEquals(new double[]{9.0},
                Cost.diffSquared.calculate(new double[]{4}, new double[]{1}), DELTA);
    }

    @Test
    void diffSquared_calculate_emptyArrays_returnEmpty() {
        // the per-element loop never runs, so the `/ input.length` divide-by-zero is never reached
        assertArrayEquals(new double[0],
                Cost.diffSquared.calculate(new double[0], new double[0]), DELTA);
    }

    @Test
    void diffSquared_derivative_knownResult() {
        // 2 * (x - y) / n  per element
        assertArrayEquals(new double[]{2.0, 0.0},
                Cost.diffSquared.derivative(new double[]{3, 1}, new double[]{1, 1}), DELTA);
    }

    @Test
    void diffSquared_derivative_perfectPredictionIsZero() {
        assertArrayEquals(new double[]{0.0},
                Cost.diffSquared.derivative(new double[]{5}, new double[]{5}), DELTA);
    }

    // ---- crossEntropy ------------------------------------------------------

    @Test
    void crossEntropy_calculate_knownResults() {
        assertArrayEquals(new double[]{-Math.log(0.8)},
                Cost.crossEntropy.calculate(new double[]{0.8}, new double[]{1}), DELTA);
        assertArrayEquals(new double[]{-Math.log(0.7)},
                Cost.crossEntropy.calculate(new double[]{0.3}, new double[]{0}), DELTA);
    }

    @Test
    void crossEntropy_calculate_zeroProbabilityForTrueLabel_throwsAssertionError() {
        // requires -ea: log(0) = -Infinity, so the cost is non-finite and the output guard fires
        assertThrows(Exception.class,
                () -> Cost.crossEntropy.calculate(new double[]{0.0}, new double[]{1}));
    }

    @Test
    void crossEntropy_calculate_unitProbabilityForFalseLabel_throwsAssertionError() {
        // requires -ea: log(1 - 1) = log(0) = -Infinity
        assertThrows(Exception.class,
                () -> Cost.crossEntropy.calculate(new double[]{1.0}, new double[]{0}));
    }

    @Test
    void crossEntropy_derivative_trueLabel_isCorrect() {
        // for y = 1 the gradient is d/dx[-log(x)] = -1/x; at x = 0.5 that is -2.0
        assertArrayEquals(new double[]{-2.0},
                Cost.crossEntropy.derivative(new double[]{0.5}, new double[]{1}), DELTA);
    }

    @Test
    void crossEntropy_derivative_falseLabel_shouldMatchAnalyticGradient() {
        // FAIL-LOUDLY: for y = 0 the gradient is d/dx[-log(1 - x)] = 1/(1 - x); at x = 0.5 that
        // is 2.0. Cost.java:50 computes `(1 - y) * (1 - x)` (multiplication) where the formula
        // needs `(1 - y) / (1 - x)` (division), so it currently returns 0.5 instead of 2.0.
        assertArrayEquals(new double[]{2.0},
                Cost.crossEntropy.derivative(new double[]{0.5}, new double[]{0}), DELTA);
    }

    // ---- contract: non-finite inputs --------------------------------------

    @Test
    void calculate_withNaNOutput_throwsAssertionError() {
        // requires -ea
        assertThrows(Exception.class,
                () -> Cost.diffSquared.calculate(new double[]{Double.NaN}, new double[]{0}));
    }

    @Test
    void derivative_withInfiniteOutput_throwsAssertionError() {
        // requires -ea
        assertThrows(Exception.class,
                () -> Cost.diffSquared.derivative(new double[]{Double.POSITIVE_INFINITY}, new double[]{0}));
    }

    @Test
    void diffSquared_calculate_swapInputAndExpected_sameCost() {
        double[] a = {3, 4};
        double[] b = {1, 2};
        assertArrayEquals(Cost.diffSquared.calculate(a, b), Cost.diffSquared.calculate(b, a), DELTA);
    }

    @Test
    void diffSquared_calculate_largePredictionError_scalesAsSquare() {
        double[] cost = Cost.diffSquared.calculate(new double[]{100}, new double[]{0});
        assertEquals(10000.0, cost[0], DELTA);
    }

    @Test
    void diffSquared_calculate_negativeValues_areHandledLikePositives() {
        // squared error is invariant under joint sign flip: (x-y)^2 == ((-x) - (-y))^2
        assertArrayEquals(Cost.diffSquared.calculate(new double[]{-3}, new double[]{-1}),
                Cost.diffSquared.calculate(new double[]{3}, new double[]{1}), DELTA);
    }

    @Test
    void diffSquared_calculate_returnsNewArray_notAlias() {
        double[] out = Cost.diffSquared.calculate(new double[]{1}, new double[]{2});
        assertNotSame(out, new double[]{1});
    }

    @Test
    void diffSquared_derivative_singleElement_knownResult() {
        assertArrayEquals(new double[]{6.0}, Cost.diffSquared.derivative(new double[]{4}, new double[]{1}), DELTA);
    }

    @Test
    void diffSquared_derivative_emptyArrays_returnEmpty() {
        assertArrayEquals(new double[0], Cost.diffSquared.derivative(new double[0], new double[0]), DELTA);
    }

    @Test
    void diffSquared_derivative_signFollowsErrorDirection() {
        double[] grad = Cost.diffSquared.derivative(new double[]{5}, new double[]{1});
        assertTrue(grad[0] > 0);
    }

    @Test
    void crossEntropy_calculate_falseLabelPerfectPrediction_isZero() {
        assertArrayEquals(new double[]{0.0}, Cost.crossEntropy.calculate(new double[]{0.0}, new double[]{0.0}), DELTA);
    }

    @Test
    void crossEntropy_calculate_trueLabelPerfectPrediction_isZero() {
        assertArrayEquals(new double[]{0.0}, Cost.crossEntropy.calculate(new double[]{1.0}, new double[]{1.0}), DELTA);
    }

    @Test
    void crossEntropy_calculate_partialProbability_matchesFormula() {
        double[] out = Cost.crossEntropy.calculate(new double[]{0.2}, new double[]{0.5});
        double expected = -(0.5 * Math.log(0.2) + 0.5 * Math.log(0.8));
        assertEquals(expected, out[0], 1e-9);
    }

    @Test
    void crossEntropy_calculate_emptyArrays_returnEmpty() {
        assertArrayEquals(new double[0], Cost.crossEntropy.calculate(new double[0], new double[0]), DELTA);
    }

    @Test
    void crossEntropy_calculate_returnsNewArray_notAlias() {
        double[] out = Cost.crossEntropy.calculate(new double[]{0.3}, new double[]{0.0});
        assertNotSame(out, new double[]{0.3});
    }

    @Test
    void crossEntropy_derivative_falseLabelOneInput_isGuardedToZero() {
        // input == 1 would yield -(0 - 1/0) = -Infinity; the `input[i] == 1 ? 0 : ...` guard
        // (mirror of the `input[i] == 0 ? 0 : ...` guard on the other term) returns 0 instead.
        assertArrayEquals(new double[]{0.0},
                Cost.crossEntropy.derivative(new double[]{1.0}, new double[]{0.0}), DELTA);
    }

    @Test
    void crossEntropy_derivative_singleElement_knownResult() {
        assertArrayEquals(new double[]{2.0}, Cost.crossEntropy.derivative(new double[]{0.5}, new double[]{0}), DELTA);
    }

    @Test
    void crossEntropy_derivative_emptyArrays_returnEmpty() {
        assertArrayEquals(new double[0], Cost.crossEntropy.derivative(new double[0], new double[0]), DELTA);
    }

    @Test
    void calculate_doesNotMutateOutputArgument() {
        double[] out = {0.3};
        double[] copy = out.clone();
        Cost.crossEntropy.calculate(out, new double[]{1.0});
        assertArrayEquals(copy, out, DELTA);
    }

    @Test
    void derivative_doesNotMutateExpectedOutputArgument() {
        double[] expected = {1.0};
        double[] copy = expected.clone();
        Cost.diffSquared.derivative(new double[]{1.0}, expected);
        assertArrayEquals(copy, expected, DELTA);
    }

    @Test
    void derivative_withNaNExpectedOutput_throwsException() {
        assertThrows(Exception.class,
                () -> Cost.diffSquared.derivative(new double[]{1.0}, new double[]{Double.NaN}));
    }

    @Test
    void calculate_inputShorterThanExpected_truncatesToInputLength() {
        // the per-element loop runs `for i in 0..input.length`, so a shorter input simply stops
        // the loop early and produces a result the size of `input`, no exception
        double[] result = Cost.diffSquared.calculate(new double[]{1}, new double[]{1, 2});
        assertEquals(1, result.length);
    }

    @Test
    void calculate_inputLengthGreaterThanExpected_throwsArrayIndexOutOfBounds() {
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> Cost.diffSquared.calculate(new double[]{1, 2}, new double[]{1}));
    }
}
