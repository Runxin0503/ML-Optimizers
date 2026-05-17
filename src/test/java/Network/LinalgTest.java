package Network;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Edge-case unit tests for {@link Linalg}, the static linear-algebra helpers.
 *
 * <p>Each test pins down one specific behaviour: a known result, an empty input, a dimension
 * mismatch, or a numeric extreme. Tests that expect an {@link AssertionError} rely on Java
 * assertions being enabled ({@code -ea}); Maven Surefire enables them by default.
 */
class LinalgTest {

    private static final double DELTA = 1e-12;

    // ---- matrixMultiply ----------------------------------------------------

    @Test
    void matrixMultiply_knownResult() {
        // 'input' indexes the rows of 'matrix': output[j] = sum_i(input[i] * matrix[i][j])
        double[][] matrix = {{1, 2, 3}, {4, 5, 6}};
        double[] input = {1, 1};
        assertArrayEquals(new double[]{5, 7, 9}, Linalg.matrixMultiply(matrix, input), DELTA);
    }

    @Test
    void matrixMultiply_weightedRows() {
        double[][] matrix = {{1, 0, 0}, {0, 1, 0}};
        double[] input = {2, 3};
        assertArrayEquals(new double[]{2, 3, 0}, Linalg.matrixMultiply(matrix, input), DELTA);
    }

    @Test
    void matrixMultiply_singleElement() {
        assertArrayEquals(new double[]{10},
                Linalg.matrixMultiply(new double[][]{{2}}, new double[]{5}), DELTA);
    }

    @Test
    void matrixMultiply_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.matrixMultiply(new double[][]{{1, 2}}, new double[]{1, 2}));
    }

    @Test
    void matrixMultiply_emptyMatrixAndVector_throwsArrayIndexOutOfBounds() {
        // the collector's supplier dereferences matrix[0] even when the stream is empty
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> Linalg.matrixMultiply(new double[0][], new double[0]));
    }

    @Test
    void matrixMultiply_nanPropagates() {
        // Linalg has no finite-value guard, so NaN flows straight through to the output
        double[] result = Linalg.matrixMultiply(new double[][]{{1.0}}, new double[]{Double.NaN});
        assertTrue(Double.isNaN(result[0]));
    }

    // ---- dotProduct --------------------------------------------------------

    @Test
    void dotProduct_knownResult() {
        assertEquals(32.0, Linalg.dotProduct(new double[]{1, 2, 3}, new double[]{4, 5, 6}), DELTA);
    }

    @Test
    void dotProduct_orthogonalVectors_isZero() {
        assertEquals(0.0, Linalg.dotProduct(new double[]{1, 0}, new double[]{0, 1}), DELTA);
    }

    @Test
    void dotProduct_emptyArrays_isZero() {
        assertEquals(0.0, Linalg.dotProduct(new double[0], new double[0]), DELTA);
    }

    @Test
    void dotProduct_singleElement() {
        assertEquals(12.0, Linalg.dotProduct(new double[]{3}, new double[]{4}), DELTA);
    }

    @Test
    void dotProduct_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.dotProduct(new double[]{1}, new double[]{1, 2}));
    }

    @Test
    void dotProduct_overflowToInfinity() {
        // no overflow handling: 1e200 * 1e200 = 1e400, which is beyond Double.MAX_VALUE
        assertTrue(Double.isInfinite(Linalg.dotProduct(new double[]{1e200}, new double[]{1e200})));
    }

    // ---- multiply ----------------------------------------------------------

    @Test
    void multiply_knownResult() {
        assertArrayEquals(new double[]{4, 10, 18},
                Linalg.multiply(new double[]{1, 2, 3}, new double[]{4, 5, 6}), DELTA);
    }

    @Test
    void multiply_emptyArrays_returnsEmpty() {
        assertArrayEquals(new double[0], Linalg.multiply(new double[0], new double[0]), DELTA);
    }

    @Test
    void multiply_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.multiply(new double[]{1, 2}, new double[]{1}));
    }

    // ---- scale / scaleInPlace ---------------------------------------------

    @Test
    void scale_knownResult() {
        assertArrayEquals(new double[]{2, 4, 6}, Linalg.scale(2, new double[]{1, 2, 3}), DELTA);
    }

    @Test
    void scale_byZero_returnsZeros() {
        assertArrayEquals(new double[]{0, 0, 0}, Linalg.scale(0, new double[]{1, -2, 3}), DELTA);
    }

    @Test
    void scale_byNegativeOne_negates() {
        assertArrayEquals(new double[]{-1, 2, -3}, Linalg.scale(-1, new double[]{1, -2, 3}), DELTA);
    }

    @Test
    void scale_byNaN_producesNaNArray() {
        // scale has no finite-value guard
        double[] result = Linalg.scale(Double.NaN, new double[]{1, 2});
        assertTrue(Double.isNaN(result[0]) && Double.isNaN(result[1]));
    }

    @Test
    void scale_emptyArray_returnsEmpty() {
        assertArrayEquals(new double[0], Linalg.scale(5, new double[0]), DELTA);
    }

    @Test
    void scaleInPlace_mutatesArgument() {
        double[] array = {1, 2, 3};
        Linalg.scaleInPlace(3, array);
        assertArrayEquals(new double[]{3, 6, 9}, array, DELTA);
    }

    @Test
    void scaleInPlace_emptyArray_isNoOp() {
        double[] array = new double[0];
        Linalg.scaleInPlace(2, array);
        assertEquals(0, array.length);
    }

    // ---- add / addInPlace --------------------------------------------------

    @Test
    void add_knownResult() {
        assertArrayEquals(new double[]{4, 6}, Linalg.add(new double[]{1, 2}, new double[]{3, 4}), DELTA);
    }

    @Test
    void add_emptyArrays_returnsEmpty() {
        assertArrayEquals(new double[0], Linalg.add(new double[0], new double[0]), DELTA);
    }

    @Test
    void add_exactCancellation() {
        assertArrayEquals(new double[]{0.0}, Linalg.add(new double[]{1e100}, new double[]{-1e100}), DELTA);
    }

    @Test
    void add_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.add(new double[]{1}, new double[]{1, 2}));
    }

    @Test
    void addInPlace_mutatesFirstArgument() {
        double[] first = {1, 2, 3};
        Linalg.addInPlace(first, new double[]{10, 20, 30});
        assertArrayEquals(new double[]{11, 22, 33}, first, DELTA);
    }

    @Test
    void addInPlace_dimensionMismatch_throwsAssertionError() {
        // requires -ea
        assertThrows(AssertionError.class,
                () -> Linalg.addInPlace(new double[]{1, 2}, new double[]{1}));
    }

    // ---- sum ---------------------------------------------------------------

    @Test
    void sum_knownResult() {
        assertEquals(6.0, Linalg.sum(new double[]{1, 2, 3}), DELTA);
    }

    @Test
    void sum_emptyArray_isZero() {
        assertEquals(0.0, Linalg.sum(new double[0]), DELTA);
    }

    @Test
    void sum_singleElement() {
        assertEquals(-7.0, Linalg.sum(new double[]{-7}), DELTA);
    }

    @Test
    void sum_withNaN_isNaN() {
        assertTrue(Double.isNaN(Linalg.sum(new double[]{1, Double.NaN, 3})));
    }
}
